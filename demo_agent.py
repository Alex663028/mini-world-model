"""MiniWorldAgent 完整演示：统一入口 + 64 维潜状态 + 生成检索混合。

覆盖五类路由：
  1. 算术（符号计算器）
  2. 任务 / 环境操控 / 边界（世界模型行动层）
  3. 感知（64 维潜状态注入：当前 + 预见）
  4. 常识（语言生成 + 检索兜底）
  5. 闲聊（语言生成）

用法：
    python demo_agent.py
    python demo_agent.py --interactive
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import MiniWorldAgent


def main():
    bot = MiniWorldAgent()
    print("=" * 58)
    print("MiniWorldAgent：世界模型 × 语言模型统一系统")
    print(f"世界模型 167 万参数 | 语言模型 75.4M（32 维潜状态注入）")
    print("=" * 58)

    pos, target = bot.env.pos, bot.env.target
    print(f"\n[环境] 我在 {pos.round(2)}，目标在 {target.round(2)}，"
          f"距离 {round(float(((pos - target) ** 2).sum() ** 0.5), 2)}")

    print("\n--- ① 感知（32 维潜状态注入）---")
    for q in ["你看到了什么？", "目标在哪个方向？", "你离目标多远？"]:
        print(f"\n老板: {q}")
        print(f"小WM: {bot.reply(q)}")

    print("\n--- ② 常识（语言生成 + 检索兜底）---")
    for q in ["光速是多少？", "柏拉图是谁？"]:
        print(f"\n老板: {q}")
        print(f"小WM: {bot.reply(q)}")

    print("\n--- ③ 算术（符号计算器）---")
    print(f"\n老板: 23 乘以 7 等于几？")
    print(f"小WM: {bot.reply('23 乘以 7 等于几？')}")

    print("\n--- ④ 任务（MPC 规划 + PD 控制，完整闭环）---")
    print(f"\n老板: 去目标那儿，导航过去")
    print(f"小WM: {bot.reply('去目标那儿，导航过去')}")

    print("\n--- ⑤ 环境操控与边界 ---")
    for q in ["墙在哪？", "重置，换个新目标"]:
        print(f"\n老板: {q}")
        print(f"小WM: {bot.reply(q)}")

    print("\n" + "=" * 58)
    print("交互模式: python demo_agent.py --interactive")


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        bot = MiniWorldAgent()
        print("MiniWorldAgent 待命中。说'去目标那儿'下任务，'重置'换目标，问感知/常识/算术都行。exit 退出")
        while True:
            try:
                q = input("\n老板: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if q.lower() in ("exit", "quit", "退出"):
                print("小WM: 再见。")
                break
            print(f"小WM: {bot.reply(q)}")
    else:
        main()
