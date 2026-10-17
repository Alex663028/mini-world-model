"""迷你世界模型实验：2D 点质量导航环境。

设计说明（原创实现，未参考任何现有仓库代码）：
- 状态空间：位置 (x, y) 与速度 (vx, vy)，均归一化到 [-1, 1] 附近
- 动作空间：二维加速度 (ax, ay)，裁剪到 [-1, 1]
- 物理：带阻尼的匀加速运动 + 边界反弹
- 观测：64x64 单通道灰度图（agent 亮点、目标灰点、运动尾迹）
- 纯 numpy 实现，不依赖 gym / box2d，Windows / Linux / macOS 直接跑
"""

import numpy as np


class PointMassEnv:
    """点质量导航环境：控制一个小球到达随机目标点。"""

    def __init__(self, size=64, max_steps=60, seed=0):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.trail_len = 6
        self.reset()

    # ---- 内部工具 ----

    def _to_pixel(self, coord):
        """把 [-1, 1] 的世界坐标映射到像素坐标。"""
        return (coord + 1.0) / 2.0 * (self.size - 8) + 4

    def _draw_disc(self, img, cx, cy, radius, value):
        """在图像上画一个实心圆盘（高斯软化边缘）。"""
        xs = np.arange(self.size)
        ys = np.arange(self.size)
        gx, gy = np.meshgrid(xs, ys)
        d2 = (gx - cx) ** 2 + (gy - cy) ** 2
        img[...] = np.maximum(img, value * np.exp(-d2 / (2.0 * (radius / 2.0) ** 2)))

    def _draw_ring(self, img, cx, cy, radius, value):
        """在图像上画一个空心圆环（中心是暗的）。

        目标点用环而不是实心盘：这样任何一帧里最亮点都唯一对应
        agent 的实心点，想象帧中也能用 argmax 可靠提取 agent 位置
        （实心目标盘会在 rollout 中抢走 argmax，实测过）。
        """
        xs = np.arange(self.size)
        ys = np.arange(self.size)
        gx, gy = np.meshgrid(xs, ys)
        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        ring = np.exp(-((d - radius) ** 2) / (2.0 * (radius / 3.0) ** 2))
        img[...] = np.maximum(img, value * ring)

    # ---- 环境接口 ----

    def reset(self):
        self.pos = self.rng.uniform(-0.6, 0.6, size=2)
        self.vel = self.rng.uniform(-0.02, 0.02, size=2)
        self.target = self.rng.uniform(-0.75, 0.75, size=2)
        self.steps = 0
        self.trail = []
        return self._obs()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # 阻尼匀加速（动作效应刻意调大：每步位移 ~0.06，是初版的 2.5 倍。
        # 位移太小的话动作信号相对潜状态尺度太弱，dynamics 会学成
        # "忽略动作"的恒等映射——实测过，8 个候选方向想象终点全挤在一起）
        self.vel = self.vel * 0.80 + action * 0.12
        self.pos = self.pos + self.vel * 0.10
        # 边界反弹（能量衰减）
        for i in range(2):
            if self.pos[i] > 1.0:
                self.pos[i] = 1.0
                self.vel[i] *= -0.5
            elif self.pos[i] < -1.0:
                self.pos[i] = -1.0
                self.vel[i] *= -0.5
        self.trail.append(self.pos.copy())
        if len(self.trail) > self.trail_len:
            self.trail.pop(0)
        self.steps += 1

        dist = float(np.linalg.norm(self.pos - self.target))
        reward = -dist
        done = dist < 0.06 or self.steps >= self.max_steps
        info = {"dist": dist, "reached": dist < 0.06}
        return self._obs(), reward, done, info

    def _obs(self):
        return self.render()

    def render(self):
        img = np.zeros((self.size, self.size), dtype=np.float32)
        # 可见边界（墙）：四周 2px 灰条。边界必须看得见——否则 encoder
        # 无法从像素感知墙的位置，想象中 agent 会"穿墙而过"，
        # MPC 被虚假想象一路骗到墙根卡死（实测过）。
        img[:2, :] = np.maximum(img[:2, :], 0.25)
        img[-2:, :] = np.maximum(img[-2:, :], 0.25)
        img[:, :2] = np.maximum(img[:, :2], 0.25)
        img[:, -2:] = np.maximum(img[:, -2:], 0.25)
        # 目标点：空心灰环（环上 0.45，中心暗）
        self._draw_ring(img, self._to_pixel(self.target[0]), self._to_pixel(self.target[1]), 4.0, 0.45)
        # 尾迹：越旧越淡
        for k, p in enumerate(self.trail):
            self._draw_disc(img, self._to_pixel(p[0]), self._to_pixel(p[1]), 2.0, 0.12 + 0.05 * k)
        # agent：白点
        self._draw_disc(img, self._to_pixel(self.pos[0]), self._to_pixel(self.pos[1]), 3.0, 1.0)
        return img

    @property
    def state(self):
        """底层真实状态（仅供 demo 对比用，不喂给世界模型）。"""
        return np.concatenate([self.pos, self.vel]).astype(np.float32)


def random_policy(rng):
    """平滑随机策略：动作带惯性，比纯白噪声更容易产生可学习的数据。"""
    action = np.zeros(2)
    while True:
        action = 0.8 * action + 0.2 * rng.uniform(-1.0, 1.0, size=2)
        yield np.clip(action, -1.0, 1.0)


def collect_rollouts(env, episodes, rng):
    """用平滑随机策略采集 (obs, action, next_obs) 转移数据。

    其中 1/3 回合使用"冲墙策略"（动作幅度不低于 0.5），专门采集
    高速撞墙/贴墙反弹的样本。随机游走数据里贴墙样本太少，世界模型
    学不会反弹动力学，想象中墙是软的，MPC 会被骗到墙根（实测过）。
    """
    policy = random_policy(rng)
    obs_buf, act_buf, nxt_buf = [], [], []
    for ep in range(episodes):
        aggressive = ep % 3 == 0
        obs = env.reset()
        done = False
        while not done:
            action = next(policy)
            if aggressive:
                action = np.sign(action) * np.maximum(np.abs(action), 0.5)
            next_obs, _, done, _ = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            nxt_buf.append(next_obs)
            obs = next_obs
    return (
        np.stack(obs_buf).astype(np.float32),
        np.stack(act_buf).astype(np.float32),
        np.stack(nxt_buf).astype(np.float32),
    )


if __name__ == "__main__":
    env = PointMassEnv(seed=42)
    obs = env.reset()
    for _ in range(60):
        obs, r, done, info = env.step([0.5, -0.3])
        if done:
            break
    print("obs shape:", obs.shape, "dtype:", obs.dtype)
    print("final dist to target:", round(info["dist"], 3))
