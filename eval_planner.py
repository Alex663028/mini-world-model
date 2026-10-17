"""规划策略 A/B：fixed8 vs continuous 候选（借鉴 lingbot-world 动作连续控制）。

不只测成功率，还测效率——continuous 候选数是 fixed8 的两倍，
如果成功率没明显提升就不划算。

用法：python eval_planner.py --seeds 16
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import MiniWorldAgent

SEEDS = [2026, 6100, 31337, 424242, 777, 1024, 555, 8888,
         31415, 27182, 16180, 14142, 17320, 22360, 26457, 28284]


def run(mode, seeds, horizon=None):
    cfg = {"mpc_candidates": mode}
    if horizon:
        cfg["aim_horizon"] = horizon
    bot = MiniWorldAgent(config=cfg)
    ok, steps, t0 = 0, [], time.time()
    for s in seeds:
        bot.env = type(bot.env)(seed=s)
        bot.env.reset()
        bot.last_z = None
        bot._prev_a = np.zeros(2, dtype=np.float32)
        r = bot.reply("去目标那儿")
        if "任务完成" in r:
            ok += 1
            try:
                steps.append(int(r.split("用 ")[1].split(" 步")[0]))
            except (IndexError, ValueError):
                pass
    return ok, (np.mean(steps) if steps else float("nan")), time.time() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--horizons", nargs="*", type=int, default=None,
                        help="额外测试的想象窗口长度")
    args = parser.parse_args()
    seeds = SEEDS[: args.seeds]

    print(f"=== 规划策略 A/B（{len(seeds)} seeds）===")
    results = {}
    for mode in ("fixed8", "continuous"):
        ok, avg, elapsed = run(mode, seeds)
        results[mode] = ok
        print(f"{mode:11s}: {ok}/{len(seeds)} 成功, 平均 {avg:.1f} 步, 耗时 {elapsed:.0f}s",
              flush=True)

    print(f"\n结论: continuous 比 fixed8 {'好' if results['continuous'] > results['fixed8'] else '不优'} "
          f"（{results['continuous'] - results['fixed8']:+d}）")

    if args.horizons:
        print(f"\n=== 想象窗口 A/B（continuous 候选）===")
        for h in args.horizons:
            ok, avg, elapsed = run("continuous", seeds, horizon=h)
            print(f"horizon={h:3d}: {ok}/{len(seeds)} 成功, 平均 {avg:.1f} 步, "
                  f"耗时 {elapsed:.0f}s", flush=True)