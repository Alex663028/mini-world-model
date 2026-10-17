"""构建三域混合语料：对话 + 知识 + 任务汇报。

背景：语言模型的能力分裂成三块——lm_state 会感知不会聊天，lm_dialog 会聊天
不会知识，lm_final2 有百科味但全是幻觉。融合的思路是训一个"文本域全能"模型：
把三个领域的语料合在一起训，让它在纯文本域（对话/知识/叙事）上不再分裂。

三个来源：
  1. 对话域：corpus_dialog.py 的 385 条手写对话
  2. 知识域：669 条常识库自动转 QA（问法模板 × 领域问句）
  3. 任务域：导航进度汇报的世界状态叙述（教它把世界"讲出来"）

注意：纯文本混合训练让模型"学会"知识表述，真正的视觉感知仍靠
lm_state 的潜状态注入通道——两者互补，不是替代。

用法：python corpus_mixed.py --out corpus_mixed.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import _iter_kb_entries
from corpus_dialog import build as build_dialog

# 知识域问句模板（按问法类型区分，避免所有知识都用同一个问法）
ASK_TEMPLATES = [
    "{k}是什么",
    "什么是{k}",
    "{k}",
    "说说{k}",
    "讲讲{k}",
    "介绍一下{k}",
    "你了解{k}吗",
    "{k}的情况是什么",
]
ASK_WHO = ["{k}是谁", "{k}是什么人", "介绍一下{k}", "{k}是谁来着"]
ASK_HOW = ["{k}是怎么来的", "{k}怎么回事", "为什么有{k}"]

# 任务域：世界状态叙述（教模型把世界模型的状态讲成自然语言）
TASK_NARR = []


def _build_task_narr():
    """世界状态叙述语料：位置/方向/距离/动作的多样化表达。"""
    dirs = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    for d in dirs:
        TASK_NARR.extend([
            f"问：我在哪个方向？\n答：我在{d}边这一带。",
            f"问：目标在哪个方向？\n答：目标在我的{d}方向。",
            f"问：往{d}走会怎样？\n答：往{d}推一把，我能往那边挪一截，离目标近一点。",
        ])
    for i, (q, a) in enumerate([
        ("我动了吗", "动了。刚才那一步推过去，画面里的我挪了位置。"),
        ("我到目标了吗", "还差一口气，最后那段我用逆动力学收着走。"),
        ("你怎么知道路怎么走", "我在世界模型里想象了 12 步，挑梦里离目标最近的那个动作。"),
        ("为什么老走偏", "想象窗口只有 12 步，长距离会累积误差，走远了我就容易晃。"),
        ("撞墙了怎么办", "撞墙会反弹，速度减半。我会换个方向再试。"),
        ("你现在在干嘛", "我编码了当前画面，把潜状态注入语言模型，正在组织语言。"),
        ("你看得见吗", "看得见。64×64 的灰度画面，我是个亮点，目标是灰环。"),
        ("你这个世界有多大", "上下左右各一个单位，撞上就反弹。"),
    ]):
        TASK_NARR.append(f"问：{q}\n答：{a}")


def build():
    """合并三域语料。"""
    parts = []

    # 1. 对话域
    parts.append(build_dialog())

    # 2. 知识域：知识库 → QA
    kb_lines = []
    for i, (keys, answer) in enumerate(_iter_kb_entries()):
        key = keys[0]
        t = ASK_TEMPLATES[i % len(ASK_TEMPLATES)]
        q = t.format(k=key)
        kb_lines.append(f"问：{q}\n答：{answer}")
        # 约三成条目再用"是谁"问法变体（适合人名/概念）
        if i % 3 == 0:
            q2 = ASK_WHO[i % len(ASK_WHO)].format(k=key)
            kb_lines.append(f"问：{q2}\n答：{answer}")
        if i % 7 == 0:
            q3 = ASK_HOW[i % len(ASK_HOW)].format(k=key)
            kb_lines.append(f"问：{q3}\n答：{answer}")
    parts.append("\n".join(kb_lines))

    # 3. 任务域
    _build_task_narr()
    parts.append("\n".join(TASK_NARR))

    return "\n\n".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="corpus_mixed.txt")
    args = parser.parse_args()
    text = build()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{args.out}: {len(text)} chars, {text.count('问：')} Q&A pairs")