"""生成"64 维潜状态 → 语言描述"配对数据（L2+ 增强版）。

相对 32 维版的升级：潜状态 = 当前帧 z（32 维）+ 动力学对下一步的预测 z_next（32 维）。
语言模型因此同时感知"现在"和"下一刻"——后 32 维教会它预见。

新增第四类问法"你接下来会怎么动？"，答案由实际执行的动作方向生成
（对应后 32 维的信息），逼语言模型学会用预见通道。

每条样本：
    text  = "问：……\n[STATE]\n答：……"
    state = 64 维（z ⊕ z_next，z_next = dynamics(z, 实际动作)）

用法：python gen_state_data.py --episodes 150
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat import extract_agent_pos
from env import PointMassEnv
from model import MiniWorldModel

STATE_TOKEN = "\uE000"
DIR8 = ["正东", "东北", "正北", "西北", "正西", "西南", "正南", "东南"]

Q_SEE = ["你看到了什么？", "描述一下你眼前的画面。", "你现在能看到什么？"]
A_SEE = [
    "画面里有个亮点是我自己（正往{move_dir}移动），在{agent_dir}侧；还有一个灰色圆环是目标点，在{target_dir}侧。",
    "我处在画面{agent_dir}侧的位置，正往{move_dir}移动；目标点在画面{target_dir}侧，是一个灰环。",
    "我的亮点在画面{agent_dir}侧，正在向{move_dir}移动，目标灰环在画面{target_dir}侧。",
]
Q_DIR = ["目标在哪个方向？", "目标在我的什么方位？", "往哪边走能找到目标？"]
A_DIR = [
    "目标在我的{dir}方向，直线距离大约{dist}个单位。",
    "往{dir}边走，大约{dist}个单位远。",
    "{dir}方向，距离约{dist}。",
]
Q_DIST = ["你离目标多远？", "距离目标还有多远？", "还差多少距离？"]
A_DIST = ["直线距离{dist}个单位。", "还有大约{dist}个单位。", "距离{dist}左右。"]
Q_NEXT = ["你接下来会怎么动？", "下一步你打算怎么做？", "你即将做什么动作？"]
A_NEXT = [
    "我会向{act_dir}方向施加推力，预计下一刻出现在画面{next_dir}附近。",
    "下一步向{act_dir}推，梦里的下一帧我在{next_dir}侧。",
    "施加{act_dir}方向的力，下一刻应该在画面{next_dir}侧。",
]


def where(p):
    x, y = p
    if abs(x) < 0.25 and abs(y) < 0.25:
        return "中间"
    parts = []
    if y < -0.2:
        parts.append("上")
    elif y > 0.2:
        parts.append("下")
    if x < -0.2:
        parts.append("左")
    elif x > 0.2:
        parts.append("右")
    return "偏" + "".join(parts) if parts else "中间"


def bearing(pos, target):
    dx, dy = target - pos
    ang = math.degrees(math.atan2(-dy, dx))
    return DIR8[int(((ang + 22.5) % 360) // 45)]


def action_dir(a):
    ang = math.degrees(math.atan2(-a[1], a[0]))
    return DIR8[int(((ang + 22.5) % 360) // 45)]


def move_dir(vel, threshold=0.02):
    """从当前速度推导运动方向（几乎静止时返回"静止"）。"""
    n = float(np.linalg.norm(vel))
    if n < threshold:
        return "静止"
    ang = math.degrees(math.atan2(-vel[1], vel[0]))
    return DIR8[int(((ang + 22.5) % 360) // 45)]


def gen_episodes(model, n_episodes, seed, use_mpc):
    env = PointMassEnv(seed=seed)
    rng = np.random.default_rng(seed)
    samples = []  # (text, z, dz) —— dz 稍后统一归一化缩放
    z_prev = None  # 上一帧潜状态（算 Δz 运动特征）

    def choose_action(obs):
        if not use_mpc:
            return rng.uniform(-1, 1, 2).astype(np.float32)
        with torch.no_grad():
            z = model.encode(torch.from_numpy(obs[None, None].astype(np.float32)))
            best_a, best_d = None, np.inf
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                a = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * 0.7
                zc = z
                for _ in range(12):
                    zc = model.imagine_step(zc, torch.from_numpy(a).unsqueeze(0))
                p = extract_agent_pos(model.decode(zc).squeeze().numpy())
                d = float(np.linalg.norm(p - env.target))
                if d < best_d:
                    best_a, best_d = a, d
        return best_a

    for ep in range(n_episodes):
        obs = env.reset()
        z_prev = None
        for step in range(60):
            action = choose_action(obs)  # 先选动作，才能算"预见"
            obs_t = torch.from_numpy(obs[None, None].astype(np.float32))
            with torch.no_grad():
                z = model.encode(obs_t)
                z_next = model.imagine_step(z, torch.from_numpy(action.astype(np.float32))[None])
                next_img = model.decode(z_next).squeeze().numpy()
                next_pos = extract_agent_pos(next_img)
                # 运动特征：当前潜状态与上一帧之差（语言模型的"速度感知"）
                dz = torch.zeros_like(z) if z_prev is None else z - z_prev
            z_prev = z.detach()

            pos, target = env.pos, env.target
            dist = round(float(np.linalg.norm(pos - target)), 1)
            agent_dir, target_dir = where(pos), where(target)
            brg = bearing(pos, target)
            act_d = action_dir(action)
            next_d = where(next_pos)
            mv_d = move_dir(env.vel)
            t = (ep * 60 + step) % 3

            for q_list, a_list, fmt in [
                (Q_SEE, A_SEE, dict(agent_dir=agent_dir, target_dir=target_dir, move_dir=mv_d)),
                (Q_DIR, A_DIR, dict(dir=brg, dist=dist)),
                (Q_DIST, A_DIST, dict(dist=dist)),
                (Q_NEXT, A_NEXT, dict(act_dir=act_d, next_dir=next_d)),
            ]:
                text = f"问：{q_list[t]}\n{STATE_TOKEN}\n答：{a_list[t].format(**fmt)}\n"
                samples.append((text, z[0].numpy(), dz[0].numpy()))

            obs, _, done, _ = env.step(action)
            if done:
                break
    return samples


def finalize(samples, z_std_target=3.0):
    """Δz 归一化：把帧间差的尺度放大到与 z 同量级。

    失败教训：Δz 原生于 std ~0.1，而 z std ~3——运动信号被位置信号淹没，
    注入通道学不动（连续两次实验失败）。缩放到同尺度后信噪比匹配。
    """
    dz_all = np.stack([s[2] for s in samples])
    dz_std = float(np.std(dz_all))
    scale = z_std_target / max(dz_std, 1e-6)
    SCALE[0] = scale  # 供 main 落盘
    out = []
    for text, z, dz in samples:
        state = np.concatenate([z, dz * scale]).tolist()
        out.append({"text": text, "state": state})
    print(f"dz std {dz_std:.4f} -> scaled x{scale:.1f} (target std {z_std_target})")
    return out


SCALE = [1.0]  # finalize 写、main 读（模块级传递缩放系数）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt")
    parser.add_argument("--out", type=str, default="state_qa_motion.jsonl")
    args = parser.parse_args()

    model = MiniWorldModel(latent_dim=32, action_dim=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    half = args.episodes // 2
    samples = gen_episodes(model, half, seed=2026, use_mpc=False)
    samples += gen_episodes(model, args.episodes - half, seed=6100, use_mpc=True)
    out = finalize(samples)  # Δz 归一化到与 z 同尺度
    print(f"generated {len(out)} state-QA samples (64-dim: z + scaled delta-z)")
    with open(args.out, "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    # 缩放系数落盘：train_lm_state 会把它存进 checkpoint config，
    # agent 推理时用同一个系数缩放 Δz——保证训练/推理分布一致
    with open("state_scale.json", "w", encoding="utf-8") as f:
        json.dump({"dz_scale": SCALE[0]}, f)
    print(f"saved -> {args.out} (dz_scale={SCALE[0]:.1f})")


if __name__ == "__main__":
    main()
