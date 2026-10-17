"""世界模型 v2：混合策略采集（随机 + MPC），结构不变、权重兼容。

改进动机（来自实测诊断）：
  v1 的动力学只见过**随机策略**的轨迹。这导致：
  - MPC 规划时"梦里"的动作分布和训练分布不匹配（想象失真）
  - 导航成功率卡在 6/8，剩余失败多发生在 MPC 方向选择阶段

v2 的做法：**数据换、模型不变**。
  - 50% 随机策略（保持对任意动作的覆盖）
  - 50% 今天调好的 MPC+PD 分层策略（让动力学在"聪明动作"分布上更准）
  - 数据量翻倍（400 → 800 回合）

关键优势：模型结构完全不变 → checkpoint 与现有 agent 兼容，
换权重不需要重训语言模型。这是一次低风险高收益的改进。

用法：
    python train_world_v2.py --episodes 800 --epochs 20 --out checkpoint_v2.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat import extract_agent_pos
from env import PointMassEnv
from model import MiniWorldModel, count_params


def mpc_action(model, env, obs, z, device, n_candidates=8, horizon=12):
    """用当前世界模型 MPC 选动作（贪心一步，评估梦里轨迹最近距离）。"""
    with torch.no_grad():
        candidates = [
            np.array([np.cos(a), np.sin(a)], dtype=np.float32) * 0.7
            for a in np.linspace(0, 2 * np.pi, n_candidates, endpoint=False)
        ]
        best_a, best_d = None, np.inf
        for a in candidates:
            zc = z
            min_d = np.inf
            for _ in range(horizon):
                zc = model.imagine_step(zc, torch.from_numpy(a).unsqueeze(0).to(device))
                p = extract_agent_pos(model.decode(zc).squeeze().cpu().numpy())
                if p is not None and np.all(np.abs(p) <= 1.05):
                    min_d = min(min_d, float(np.linalg.norm(p - env.target)))
            if min_d < best_d:
                best_a, best_d = a, min_d
    return best_a


def pd_action(env):
    """PD 逆动力学精密进近（距离小于 0.4 时用）。"""
    vec = env.target - env.pos
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return np.zeros(2, dtype=np.float32)
    desired_v = vec / n * min(0.3, n * 1.2)
    a = (desired_v - 0.8 * env.vel) / 0.12
    return np.clip(a, -0.9, 0.9).astype(np.float32)


def collect_mixed(env, episodes, rng, model, device, mpc_ratio=0.5):
    """混合策略采集：一半随机动作，一半 MPC+PD 策略。"""
    obs_l, act_l, nxt_l = [], [], []
    reached = 0
    for _ in range(episodes):
        obs = env.reset()
        for _ in range(env.max_steps):
            d = float(np.linalg.norm(env.pos - env.target))
            if rng.random() < mpc_ratio:
                obs_t = torch.from_numpy(obs[None, None].astype(np.float32)).to(device)
                with torch.no_grad():
                    z = model.encode(obs_t)
                a = pd_action(env) if d <= 0.4 else mpc_action(model, env, obs, z, device)
            else:
                a = rng.uniform(-1, 1, 2).astype(np.float32)
            obs_t = torch.from_numpy(obs[None, None].astype(np.float32)).to(device)
            with torch.no_grad():
                z_next = model.imagine_step(model.encode(obs_t),
                                            torch.from_numpy(a).unsqueeze(0).to(device))
                nxt = model.decode(z_next).squeeze().cpu().numpy()  # 想象的下一帧
            obs_l.append(obs)
            act_l.append(a)
            nxt_l.append(nxt)
            obs, _, done, info = env.step(a)
            if done:
                reached += 1
                break
    return (np.array(obs_l, dtype=np.float32),
            np.array(act_l, dtype=np.float32),
            np.array(nxt_l, dtype=np.float32),
            reached)


def collect_random(env, episodes, rng):
    """随机策略采集（保持对任意动作的覆盖）。"""
    obs_l, act_l, nxt_l = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        for _ in range(env.max_steps):
            a = rng.uniform(-1, 1, 2).astype(np.float32)
            obs_l.append(obs)
            act_l.append(a)
            obs, _, done, _ = env.step(a)
            nxt_l.append(env.render())
            if done:
                break
    return (np.array(obs_l, dtype=np.float32),
            np.array(act_l, dtype=np.float32),
            np.array(nxt_l, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800, help="随机策略回合数")
    parser.add_argument("--mpc-episodes", type=int, default=400, help="MPC 策略回合数")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="checkpoint.pt",
                        help="从 v1 权重继续训练（warm start）")
    parser.add_argument("--out", type=str, default="checkpoint_v2.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型（结构不变，保证 checkpoint 兼容）
    model = MiniWorldModel(latent_dim=32, action_dim=2).to(device)
    if os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device)
        if sd.keys() == model.state_dict().keys() and all(
                sd[k].shape == model.state_dict()[k].shape for k in sd):
            model.load_state_dict(sd)
            print(f"warm start from {args.resume}")
        else:
            print(f"{args.resume} 结构不匹配，从零训练")
    print(f"model params: {count_params(model)}, device: {device}")

    # 采集：随机 + MPC（MPC 部分用当前模型采集，训练中会越来越准）
    env = PointMassEnv(seed=args.seed)
    t0 = time.time()
    r_obs, r_act, r_nxt = collect_random(env, args.episodes, rng)
    print(f"random: {len(r_obs)} transitions ({time.time() - t0:.1f}s)")

    t1 = time.time()
    m_obs, m_act, m_nxt, reached = collect_mixed(
        PointMassEnv(seed=args.seed + 1000), args.mpc_episodes, rng, model, device)
    print(f"mpc: {len(m_obs)} transitions, 策略到达率 {reached}/{args.mpc_episodes} "
          f"({time.time() - t1:.1f}s)")

    obs = np.concatenate([r_obs, m_obs])
    act = np.concatenate([r_act, m_act])
    nxt = np.concatenate([r_nxt, m_nxt])
    print(f"total: {len(obs)} transitions (random {len(r_obs)} + mpc {len(m_obs)})")

    # 训练
    ds = TensorDataset(
        torch.from_numpy(obs).unsqueeze(1),
        torch.from_numpy(act),
        torch.from_numpy(nxt).unsqueeze(1),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        tot_rec, tot_dyn, nb = 0.0, 0.0, 0
        for ob, ac, nx in dl:
            ob, ac, nx = ob.to(device), ac.to(device), nx.to(device)
            z = model.encode(ob)
            rec = torch.nn.functional.mse_loss(model.decode(z), ob)
            z_next_pred = model.dynamics(z, ac)
            z_next_true = model.encode(nx).detach()
            dyn = torch.nn.functional.mse_loss(z_next_pred, z_next_true)
            loss = rec + dyn
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_rec += rec.item()
            tot_dyn += dyn.item()
            nb += 1
        avg = (tot_rec + tot_dyn) / nb
        print(f"epoch {epoch}/{args.epochs}  rec={tot_rec / nb:.4f}  "
              f"dyn={tot_dyn / nb:.4f}  total={avg:.4f}", flush=True)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.out)
    print(f"done. best {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()