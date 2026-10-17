"""世界模型 A/B 评测：v1（随机数据）vs v2（混合策略数据）。

三项指标：
  1. 导航成功率（agent._run_navigation，8 个 seed）
  2. 平均步数（到达的回合）
  3. 潜状态兼容性（v2 的 z 喂给现有 lm_state，感知回答是否还对）

用法：python eval_world.py --a checkpoint.pt --b checkpoint_v2.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import MiniWorldAgent


def eval_navigation(ckpt, seeds, dialog_ckpt="lm_unified.pt"):
    """跑 n 个 seed 的导航任务，统计成功率与步数。"""
    bot = MiniWorldAgent(world_ckpt=ckpt, dialog_ckpt=dialog_ckpt)
    ok, steps = 0, []
    for s in seeds:
        bot.env = type(bot.env)(seed=s)
        bot.env.reset()
        bot.last_z = None
        r = bot.reply("去目标那儿")
        if "任务完成" in r:
            ok += 1
            try:
                steps.append(int(r.split("用 ")[1].split(" 步")[0]))
            except (IndexError, ValueError):
                pass
    return ok, steps


def eval_perception(ckpt, dialog_ckpt="lm_state.pt"):
    """潜状态兼容性：换世界模型后感知回答是否还对。"""
    bot = MiniWorldAgent(world_ckpt=ckpt, dialog_ckpt=dialog_ckpt)
    bot.env = type(bot.env)(seed=2026)
    bot.env.reset()
    bot.last_z = None
    answers = []
    for q in ["你看到了什么？", "目标在哪个方向？", "你离目标多远？"]:
        answers.append(bot.reply(q))
    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="checkpoint.pt", help="基线（v1）")
    parser.add_argument("--b", default="checkpoint_v2.pt", help="候选（v2）")
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    seeds = [2026, 6100, 31337, 424242, 777, 1024, 555, 8888, 31415, 27182, 16180, 14142, 17320, 22360, 26457, 28284][: args.seeds]

    print("=== 导航成功率 A/B ===")
    a_ok, a_steps = eval_navigation(args.a, seeds)
    print(f"A ({args.a}): {a_ok}/{len(seeds)} 成功，平均步数 "
          f"{np.mean(a_steps) if a_steps else float('nan'):.1f}")
    if os.path.exists(args.b):
        b_ok, b_steps = eval_navigation(args.b, seeds)
        print(f"B ({args.b}): {b_ok}/{len(seeds)} 成功，平均步数 "
              f"{np.mean(b_steps) if b_steps else float('nan'):.1f}")
        delta = b_ok - a_ok
        print(f"\n结论: {'B 更优' if delta > 0 else ('持平' if delta == 0 else 'A 更优')} "
              f"（成功率差 {delta:+d}）")
    else:
        print(f"B 不存在（{args.b}），跳过")

    print("\n=== 潜状态兼容性（感知回答是否受影响）===")
    print("A:", eval_perception(args.a))
    if os.path.exists(args.b):
        print("B:", eval_perception(args.b))