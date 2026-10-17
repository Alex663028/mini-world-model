"""训练最小世界模型。

流程（和真实世界模型训练同构，只是规模缩小）：
  1. 用随机策略在环境里采集 (obs_t, action_t, obs_{t+1}) 转移数据
  2. 联合训练：
     - 重建损失：Decoder(Encoder(obs)) ≈ obs
     - 动力学损失：Dynamics(z_t, a_t) ≈ Encoder(obs_{t+1}).detach()
  3. 保存 checkpoint

用法：
    python train.py --episodes 400 --epochs 12
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from env import PointMassEnv, collect_rollouts
from model import MiniWorldModel, count_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400, help="随机策略回合数")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoint.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 1. 采集数据
    env = PointMassEnv(seed=args.seed)
    t0 = time.time()
    obs, act, nxt = collect_rollouts(env, args.episodes, rng)
    print(f"collected {len(obs)} transitions in {time.time() - t0:.1f}s  obs={obs.shape}")

    # 2. 训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = MiniWorldModel(latent_dim=32, action_dim=2).to(device)
    print("model params:", count_params(model))

    ds = TensorDataset(
        torch.from_numpy(obs).unsqueeze(1),
        torch.from_numpy(act),
        torch.from_numpy(nxt).unsqueeze(1),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_recon, tot_dyn, nb = 0.0, 0.0, 0
        for b_obs, b_act, b_nxt in dl:
            b_obs, b_act, b_nxt = b_obs.to(device), b_act.to(device), b_nxt.to(device)
            z = model.encode(b_obs)
            recon = model.decode(z)
            # 前景加权 BCE：agent/目标像素只占约 1%，用普通 MSE 会让模型
            # 坍缩到"输出全黑"（全黑的 MSE 已低于学画亮点的初期损失）。
            # 注意：加权只作用于接近全亮的前景（agent 实心点），
            # 半亮像素（目标环 0.45、边界 0.25）不能加权——否则 15 倍权重
            # 会逼模型把它们预测成 0.83，重建被扭曲（实测 recon 劣化 3 倍）。
            p = recon.clamp(1e-6, 1 - 1e-6)
            w_pos = 15.0 * (b_obs > 0.5).float()
            bce = -(w_pos * b_obs * torch.log(p) + (1.0 - b_obs) * torch.log(1.0 - p))
            recon_loss = bce.mean()

            with torch.no_grad():
                z_next_true = model.encode(b_nxt)  # 语义目标：下一帧的潜状态
            # 关键稳定性设计：dynamics 的输入 detach。
            # 联合训练时如果让 dynamics 的梯度流回 encoder，encoder 会被两个
            # 损失来回拉扯、表征持续漂移，dynamics 的输入分布随之突变，
            # 误差爆炸后再把 encoder 打爆，形成正反馈雪崩（实测 4 轮后 dyn 破万）。
            # detach 后 encoder 只由重建损失塑造，dynamics 在固定表征上学习。
            z_next_pred = model.imagine_step(z.detach(), b_act)
            dyn_loss = torch.mean((z_next_pred - z_next_true) ** 2)

            loss = recon_loss + dyn_loss
            opt.zero_grad()
            loss.backward()
            # 梯度裁剪：加权 BCE 后期梯度会变猛，不裁会在 20 轮后突然发散
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot_recon += recon_loss.item()
            tot_dyn += dyn_loss.item()
            nb += 1
        epoch_loss = tot_recon / nb + tot_dyn / nb
        print(f"epoch {epoch:3d}/{args.epochs}  recon={tot_recon / nb:.6f}  dyn={tot_dyn / nb:.6f}")
        # 只保存历史最佳权重（最后一轮可能已发散，不能直接用）
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.out)
            print(f"           -> best so far, saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
