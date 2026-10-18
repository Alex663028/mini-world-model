"""世界模型 v3：加显式状态预测头（LingBot-World 2.0 "完整世界状态推演"落地）。

v3 相比 v2 的改动：
  - model 增加 state_head：z -> (pos2, vel2, target2, dist, steps_norm, reached)
  - 训练 loss = 重建 + 动力学 + 0.5 × 状态回归
  - 数据采集时同步记录状态真值

为什么这很重要：
  1. 之前"状态"是解码图像 + argmax 抠出来的——慢（每步一次 CNN 解码）且粗
  2. 加了状态头，世界模型能直接"推演状态"而不只是"生成像素"，
     MPC 想象时可以只跑 MLP（快两个数量级）

注意：新增参数会破坏 checkpoint 兼容，warm start 时按形状匹配加载。

用法：
    python train_world_v3.py --episodes 400 --epochs 20 --out checkpoint_v3.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import PointMassEnv
from model import MiniWorldModel, count_params


def state_vector(env):
    """世界状态真值（9 维，与 model.STATE_DIM 对齐）。"""
    dist = float(np.linalg.norm(env.pos - env.target))
    return np.array([
        env.pos[0], env.pos[1],
        env.vel[0], env.vel[1],
        env.target[0], env.target[1],
        dist,
        env.steps / env.max_steps,
        1.0 if dist < 0.06 else 0.0,
    ], dtype=np.float32)


def collect(env, episodes, rng):
    """随机策略采集：(obs, action, next_obs, state)。"""
    obs_l, act_l, nxt_l, st_l = [], [], [], []
    for _ in range(episodes):
        obs = env.reset()
        for _ in range(env.max_steps):
            a = rng.uniform(-1, 1, 2).astype(np.float32)
            st = state_vector(env)
            obs_l.append(obs)
            act_l.append(a)
            obs, _, done, _ = env.step(a)
            nxt_l.append(env.render())
            st_l.append(st)
            if done:
                break
    return (np.array(obs_l, dtype=np.float32), np.array(act_l, dtype=np.float32),
            np.array(nxt_l, dtype=np.float32), np.array(st_l, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="checkpoint_v2.pt")
    parser.add_argument("--out", type=str, default="checkpoint_v3.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniWorldModel(latent_dim=32, action_dim=2).to(device)
    print(f"model params: {count_params(model)}, device: {device}")

    # warm start：老 checkpoint 没有 state_head，按形状匹配加载
    if os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        cur = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
                loaded += 1
        model.load_state_dict(cur)
        print(f"warm start from {args.resume}: {loaded}/{len(cur)} tensors "
              f"（state_head 是新参数，从零学）")

    env = PointMassEnv(seed=args.seed)
    t0 = time.time()
    obs, act, nxt, st = collect(env, args.episodes, rng)
    print(f"collected {len(obs)} transitions ({time.time() - t0:.1f}s)")

    ds = TensorDataset(
        torch.from_numpy(obs).unsqueeze(1),
        torch.from_numpy(act),
        torch.from_numpy(nxt).unsqueeze(1),
        torch.from_numpy(st),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        tot_rec = tot_dyn = tot_st = 0.0
        nb = 0
        for ob, ac, nx, st_t in dl:
            ob, ac, nx, st_t = ob.to(device), ac.to(device), nx.to(device), st_t.to(device)
            z = model.encode(ob)
            rec = torch.nn.functional.mse_loss(model.decode(z), ob)
            z_next_pred = model.dynamics(z, ac)
            z_next_true = model.encode(nx).detach()
            dyn = torch.nn.functional.mse_loss(z_next_pred, z_next_true)
            # 状态回归：从当前 z 预测当前状态。
            # 关键：z.detach()——状态头只是"读取器"，不能让它的梯度
            # 反传去改潜空间。第一版没 detach，state loss 的梯度冲垮了
            # 已学好的 encoder（dyn loss 从 1e-4 爆炸到 11496）。
            pred = model.state_head(z.detach())
            st_loss = torch.nn.functional.mse_loss(pred, st_t)
            loss = rec + dyn + args.state_weight * st_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_rec += rec.item()
            tot_dyn += dyn.item()
            tot_st += st_loss.item()
            nb += 1
        avg = (tot_rec + tot_dyn + args.state_weight * tot_st) / nb
        print(f"epoch {epoch}/{args.epochs}  rec={tot_rec / nb:.4f}  "
              f"dyn={tot_dyn / nb:.4f}  state={tot_st / nb:.4f}  total={avg:.4f}",
              flush=True)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.out)
    print(f"done. best {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()