"""最小世界模型演示：验证它是否真的"学会了世界"。

三个实验：
  1. 一步预测：obs_t + a_t -> 想象 obs_{t+1}，和真实下一帧对比
  2. 开环想象 rollout：完全不碰环境，用动力学迭代 30 步，
     把想象的潜状态解码成图，从中提取 agent 位置，对比真实轨迹
  3. MPC 规划：每步枚举 8 个候选动作方向，用世界模型想象 12 步，
     谁想象的终点离目标近就执行谁——看智能体能否只靠"想象"导航到目标

用法：python demo.py --ckpt checkpoint.pt
"""

import argparse

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import PointMassEnv
from model import MiniWorldModel

LATENT_DIM = 32
IMG_SIZE = 64


def load_model(ckpt_path):
    model = MiniWorldModel(latent_dim=LATENT_DIM, action_dim=2)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def to_tensor(img):
    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def extract_agent_pos(img):
    """从渲染图中提取 agent（最亮点）的世界坐标。"""
    idx = np.unravel_index(np.argmax(img), img.shape)
    y, x = idx
    wx = (x - 4) / (IMG_SIZE - 8) * 2.0 - 1.0
    wy = (y - 4) / (IMG_SIZE - 8) * 2.0 - 1.0
    return np.array([wx, wy])


def imagine_rollout(model, z0, actions):
    """开环想象：从潜状态 z0 出发，按动作序列迭代动力学。"""
    z = z0
    zs = []
    with torch.no_grad():
        for a in actions:
            z = model.imagine_step(z, torch.from_numpy(a.astype(np.float32)).unsqueeze(0))
            zs.append(z)
    return zs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = load_model(args.ckpt)
    print("model loaded")

    # ============ 实验 1：一步预测 ============
    env = PointMassEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    obs = env.reset()
    warmup_actions = []
    for _ in range(12):  # 热身几步，让画面里有尾迹
        a = rng.uniform(-1, 1, 2)
        warmup_actions.append(a)
        obs, _, done, _ = env.step(a)
        if done:
            obs = env.reset()
            warmup_actions = []
    action = np.array([0.6, -0.4], dtype=np.float32)
    real_next, _, _, _ = env.step(action)

    with torch.no_grad():
        z = model.encode(to_tensor(obs))
        z_next = model.imagine_step(z, torch.from_numpy(action).unsqueeze(0))
        imagined_next = model.decode(z_next).squeeze().numpy()

    # ============ 实验 2：开环想象 rollout ============
    horizon = 30
    smooth = np.zeros(2)
    actions = []
    for _ in range(horizon):
        smooth = 0.8 * smooth + 0.2 * rng.uniform(-1, 1, 2)
        actions.append(np.clip(smooth, -1, 1))
    actions = np.stack(actions).astype(np.float32)

    z0 = model.encode(to_tensor(obs))
    zs = imagine_rollout(model, z0, actions)
    with torch.no_grad():
        imagined_frames = [model.decode(z).squeeze().numpy() for z in zs]
    imagined_pos = np.stack([extract_agent_pos(f) for f in imagined_frames])

    real_pos = [env.state[:2].copy()]
    for a in actions:
        env.step(a)
        real_pos.append(env.state[:2].copy())
    real_pos = np.stack(real_pos)

    rollout_err = np.linalg.norm(imagined_pos - real_pos[1:], axis=1)

    # ============ 实验 3：MPC 规划 ============
    env2 = PointMassEnv(seed=args.seed + 100)
    obs2 = env2.reset()
    target = env2.target.copy()
    mpc_pos = [env2.pos.copy()]
    reached = False
    done = False
    prev_a = np.zeros(2, dtype=np.float32)
    with torch.no_grad():
        # 每个动作执行 2 步：agent 有物理惯性，每步都换方向会抖动，
        # 持续 2 步运动更连贯（也是真实 MPC 的 standard 做法）
        for _ in range(0, 60, 2):
            z = model.encode(to_tensor(obs2))
            candidates = [
                np.array([np.cos(a), np.sin(a)], dtype=np.float32) * 0.7
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
            ]
            candidates.append(prev_a.copy())                # 惯性候选：延续上一步
            candidates.append(np.zeros(2, dtype=np.float32))  # 减速候选
            best_a, best_d = None, np.inf
            for a in candidates:
                zc = z
                for _ in range(12):
                    zc = model.imagine_step(zc, torch.from_numpy(a).unsqueeze(0))
                p = extract_agent_pos(model.decode(zc).squeeze().numpy())
                d = float(np.linalg.norm(p - target))
                if d < best_d:
                    best_a, best_d = a, d
            for _ in range(2):
                obs2, _, done, info = env2.step(best_a)
                mpc_pos.append(env2.pos.copy())
                if done:
                    reached = info["reached"]
                    break
            prev_a = best_a
            if done:
                break
    mpc_pos = np.stack(mpc_pos)
    print(f"MPC: {'REACHED target' if reached else 'not reached'} in {len(mpc_pos) - 1} steps, "
          f"final dist={np.linalg.norm(env2.pos - target):.3f}")

    # ============ 可视化 ============
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle("mini world model: predict / imagine / plan", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(obs, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("current frame obs_t")
    axes[0, 1].imshow(real_next, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("real next frame (truth)")
    axes[0, 2].imshow(imagined_next, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("imagined next frame (model)")

    axes[1, 0].plot(real_pos[:, 0], real_pos[:, 1], "o-", ms=3, label="real")
    axes[1, 0].plot(imagined_pos[:, 0], imagined_pos[:, 1], "s--", ms=3, label="imagined")
    axes[1, 0].scatter(*target, marker="*", s=160, label="target")
    axes[1, 0].set_title(f"open-loop imagination ({horizon} steps)")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_xlim(-1.05, 1.05)
    axes[1, 0].set_ylim(-1.05, 1.05)

    axes[1, 1].plot(np.arange(1, horizon + 1), rollout_err, "o-", ms=3, color="#BA7517")
    axes[1, 1].set_title("imagination error vs horizon")
    axes[1, 1].set_xlabel("imagination step")
    axes[1, 1].set_ylabel("position error")

    axes[1, 2].plot(mpc_pos[:, 0], mpc_pos[:, 1], "o-", ms=3, color="#1D9E75", label="MPC path")
    axes[1, 2].scatter(*target, marker="*", s=160, color="#993C1D", label="target")
    axes[1, 2].scatter(*mpc_pos[0], marker="o", s=60, facecolors="none", edgecolors="k", label="start")
    axes[1, 2].set_title(f"MPC navigation: {'reached' if reached else 'failed'}")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].set_xlim(-1.05, 1.05)
    axes[1, 2].set_ylim(-1.05, 1.05)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "demo_result.png"
    plt.savefig(out, dpi=110)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
