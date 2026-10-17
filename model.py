"""最小世界模型：编码器 + 动力学 + 解码器。

架构（对标 World Models / RSSM 的迷你版，全部原创实现）：

  观测图像 (1x64x64)
        |
        v
  [Encoder CNN] ---> z_t (32 维潜状态)
        |                |
        |                +--- action a_t ---> [Dynamics MLP] ---> z_{t+1} 预测
        |                                        |
        v                                        v
  [Decoder CNN] <-------------------------- z_{t+1}（可解码成想象的下一帧）

训练信号：
  1. 重建损失：Decoder(Encoder(obs)) ≈ obs，逼编码器保留视觉信息
  2. 动力学损失：Dynamics(z_t, a_t) ≈ Encoder(obs_{t+1}).detach()
     —— 这就是"世界模型"本体：给定当前潜状态和动作，预测下一潜状态
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """64x64 灰度图 -> 32 维潜状态。"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """32 维潜状态 -> 重建 64x64 灰度图。"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)


class Dynamics(nn.Module):
    """(潜状态, 动作) -> 预测下一潜状态（残差形式，稳定好训）。"""

    def __init__(self, latent_dim=32, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        return z + self.net(x)


class MiniWorldModel(nn.Module):
    """三件套打包：编码、预测、解码，外加显式状态预测头。

    状态头（LingBot-World 2.0 "完整世界状态推演"的最小实现）：
    从潜状态直接回归世界状态量（位置/速度/目标/距离/步数），
    而不是靠解码图像 + argmax 抠位置。两个好处：
      1. 真正"推演状态"而非"生成像素"
      2. MPC 想象时可以只跑 MLP 状态头，不用每步解码图像（快两个数量级）
    """

    STATE_DIM = 9  # pos(2) + vel(2) + target(2) + dist(1) + steps_norm(1) + reached_flag(1)

    def __init__(self, latent_dim=32, action_dim=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.dynamics = Dynamics(latent_dim, action_dim)
        self.decoder = Decoder(latent_dim)
        # 显式状态头：z -> (pos2, vel2, target2, dist, steps_norm, reached)
        self.state_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.STATE_DIM),
        )

    def encode(self, obs):
        return self.encoder(obs)

    def imagine_step(self, z, action):
        """在想象空间前进一步：给定潜状态和动作，返回预测的下一潜状态。"""
        return self.dynamics(z, action)

    def decode(self, z):
        return self.decoder(z)

    def predict_state(self, z):
        """从潜状态推演世界状态：pos, vel, target, dist, steps_norm, reached_logit。"""
        s = self.state_head(z)
        return {
            "pos": s[:, :2],
            "vel": s[:, 2:4],
            "target": s[:, 4:6],
            "dist": s[:, 6],
            "steps_norm": s[:, 7],
            "reached_logit": s[:, 8],
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MiniWorldModel()
    print("total params:", count_params(m))
    obs = torch.randn(4, 1, 64, 64)
    act = torch.randn(4, 2)
    z = m.encode(obs)
    z_next = m.imagine_step(z, act)
    rec = m.decode(z)
    print("latent:", tuple(z.shape), "-> next latent:", tuple(z_next.shape), "-> recon:", tuple(rec.shape))
