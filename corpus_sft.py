"""构建 MiniMind 格式的 SFT 数据集（conversations JSONL）。

参考 jingyaogong/minimind 的数据规范：
- 多轮 conversations 格式（role: system/user/assistant/tool）
- Tool Calling：assistant 发 tool_calls，tool 返回结果，assistant 总结
- pretrain 与 sft 分离：本文件只产出 sft 数据（pretrain 已有 lm_final2）

三类样本：
  1. 多轮对话（含 system prompt 交代身份）
  2. 知识问答（从 669 条常识库自动生成）
  3. Tool Call 轨迹（教模型自己决定调哪个工具：感知/导航/检索/计算）

用法：python corpus_sft.py --out sft_data.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import _iter_kb_entries
from corpus_dialog import BOUND_Q, CAN_Q, FAIL_Q, GREET_Q, NO_Q, THANKS_Q, WHO_Q, YES_Q

SYSTEM = (
    "你是小WM，一个由世界模型和语言模型组成的迷你 agent。"
    "你能看见 64×64 灰度画面里的自己和目标环、能规划路径走过去、"
    "能查询常识、能做四则运算。需要这些能力时，你会调用对应工具。"
)

# ---------------- Tool Call 触发问法变体（扩量用）----------------
TOOL_ASKS = {
    "world_perceive": [
        "你看到了什么？", "描述一下画面", "眼前有什么", "现在能看见什么",
        "画面里有什么东西", "给我看看现在的场景", "你现在看到的是什么",
        "观测到什么", "你的视野里有什么", "看看四周",
    ],
    "world_navigate": [
        "去目标那儿", "走到目标", "过去", "过去看看", "去灰环那里",
        "靠近目标一点", "过去那边", "走到我说的地方", "过去看看那边",
        "往那边走", "去找目标", "去目标那儿", "帮我过去",
    ],
    "world_imagine": [
        "想象一下往左走", "想象往右走", "要是往前走会怎样", "预演一下往左",
        "想象一下往右走会怎样", "推演一下", "想象一下", "yy一下会怎样",
    ],
    "world_act": [
        "往前走一步", "往左走", "往右走", "动一下", "推一下",
    ],
}

# ---------------- Tool Call 轨迹（MiniMind 式）----------------
TOOL_TRAJ = []


def _tool_trajs():
    """工具调用轨迹：user → assistant(tool_calls) → tool(result) → assistant(答案)。

    扩量版：感知/导航/想象各用多种问法触发（模型要学会"什么问法调什么工具"，
    而不是死记一种问法）。检索/计算按知识库和算式枚举。
    """
    # 1. 感知工具（10 种问法 × 2 种回答风格）
    perceive_answers = [
        "画面里有个亮点是我自己，右边那个灰环是目标。",
        "我看到一个亮点和一个灰环，亮点是我，灰环是目标点。",
    ]
    for ask in TOOL_ASKS["world_perceive"]:
        TOOL_TRAJ.append([
            {"role": "user", "content": ask},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(
                [{"name": "world_perceive", "arguments": {}}], ensure_ascii=False)},
            {"role": "tool", "content": "画面 64×64 灰度：我是一个亮点，目标是灰色圆环。"
                                        "位置与方位可从潜状态解出。"},
            {"role": "assistant", "content": perceive_answers[len(ask) % 2]},
        ])
    # 2. 导航工具（12 种问法）
    nav_answers = [
        "任务完成！用 8 步走到了目标跟前。",
        "过去了，MPC 规划了几步，控制层收尾。",
        "到目标了，中途切换过一次控制层。",
    ]
    for ask in TOOL_ASKS["world_navigate"]:
        TOOL_TRAJ.append([
            {"role": "user", "content": ask},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(
                [{"name": "world_navigate", "arguments": {"task": ask}}], ensure_ascii=False)},
            {"role": "tool", "content": "MPC 规划 + PD 控制执行完成。"},
            {"role": "assistant", "content": nav_answers[len(ask) % 3]},
        ])
    # 3. 检索工具（知识库条目 × 多种问法）
    kb_items = list(_iter_kb_entries())
    for i, (keys, answer) in enumerate(kb_items):
        key = keys[0]
        asks = [f"{key}是什么", f"什么是{key}", f"说说{key}", f"{key}",
                f"介绍一下{key}", f"{key}的情况", f"你了解{key}吗"]
        for j, q in enumerate(asks):
            if (i + j) % 3 != 0:      # 每条知识取 2-3 种问法，控制总量
                continue
            TOOL_TRAJ.append([
                {"role": "user", "content": q},
                {"role": "assistant", "content": "", "tool_calls": json.dumps(
                    [{"name": "kb_query", "arguments": {"question": q}}], ensure_ascii=False)},
                {"role": "tool", "content": answer},
                {"role": "assistant", "content": answer},
            ])
    # 4. 计算工具（枚举算式）
    ops = [("乘以", "*"), ("加上", "+"), ("减去", "-"), ("除以", "/")]
    nums = [(23, 7), (144, 12), (98, 2), (7, 9), (56, 8), (81, 9), (64, 8), (35, 5), (12, 4), (100, 5)]
    for a, b in nums:
        for opname, _ in ops[:2] if (a + b) % 2 else ops:
            res = a * b if opname == "乘以" else (a + b if opname == "加上" else
                                                   a - b if opname == "减去" else
                                                   (a / b if b else 0))
            expr = f"{a} {opname} {b}"
            TOOL_TRAJ.append([
                {"role": "user", "content": expr},
                {"role": "assistant", "content": "", "tool_calls": json.dumps(
                    [{"name": "calc", "arguments": {"expression": expr}}], ensure_ascii=False)},
                {"role": "tool", "content": f"{expr} = {res}"},
                {"role": "assistant", "content": f"{expr} = {res}。"},
            ])
    # 5. 想象工具
    img_answers = [
        "想象里我往那边挪了一截，离目标近了点。",
        "梦里那一步会有点偏，但方向没错。",
    ]
    for ask in TOOL_ASKS["world_imagine"]:
        TOOL_TRAJ.append([
            {"role": "user", "content": ask},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(
                [{"name": "world_imagine", "arguments": {"direction": "左"}}], ensure_ascii=False)},
            {"role": "tool", "content": "12 步想象：梦里我移动到 (-0.21, -0.29)，离目标 1.29。"},
            {"role": "assistant", "content": img_answers[len(ask) % 2]},
        ])


# ---------------- 多轮对话 ----------------
def _dialogs():
    """多轮对话：系统 prompt + 身份/能力/礼貌/场景，2-3 轮追问。

    扩量：base 对话做两两组合（不只相邻），避免模型学成固定顺序。
    """
    out = []
    base = [
        ("你好", "嗨，我在呢。想让我看画面、导航，还是随便聊两句？"),
        ("你是谁", "我是小WM，由一个世界模型和一个语言模型组成。世界模型让我看得见、想得出，语言模型让我说得出。"),
        ("你能做什么", "我能感知画面、规划导航、查询常识、做四则运算。需要时我会调工具。"),
        ("谢谢", "不客气。"),
        ("再见", "回见，世界模型我给你留着。"),
        ("你多大了", "刚训出来，按训练轮数算，今天才出生。"),
        ("你会犯错吗", "会。距离估计会飘，精确数字记不住。"),
    ]
    # 两两组合成多轮
    n = len(base)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u1, a1 = base[i]
            u2, a2 = base[j]
            out.append([
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ])
    # 三轮：身份 → 能力 → 追问
    for q, a in BOUND_Q:
        out.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
            {"role": "user", "content": "那你的能力里哪一项最靠谱？"},
            {"role": "assistant", "content": "四则运算最靠谱，交给符号计算器。感知也还行，方位判断挺准。"},
        ])
    # 自我认知（去掉模板化前缀，避免"（认真回答）"过拟合）
    for q in WHO_Q + CAN_Q + GREET_Q + THANKS_Q + YES_Q + NO_Q:
        out.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": "我是小WM，一个世界模型加语言模型的迷你 agent。"},
        ])
    return out


def _knowledge():
    """知识问答（纯文本，不走工具）：每条知识多个问法，覆盖真实问法分布。"""
    out = []
    asks = ["{k}是什么", "什么是{k}", "说说{k}", "介绍一下{k}", "{k}",
            "{k}的情况是什么", "你了解{k}吗", "讲讲{k}"]
    for i, (keys, answer) in enumerate(_iter_kb_entries()):
        k = keys[0]
        n_ask = 3 if i % 2 == 0 else 2      # 3-4 种问法轮换，控制总量
        for j in range(n_ask):
            q = asks[(i + j) % len(asks)].format(k=k)
            out.append([
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ])
    for q, a in FAIL_Q:
        out.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ])
    return out


def build():
    """产出 SFT JSONL。每条 {"conversations": [...]}。"""
    _tool_trajs()
    samples = []
    for conv in _dialogs() + _knowledge() + TOOL_TRAJ:
        samples.append({"conversations": [{"role": "system", "content": SYSTEM}] + conv})
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="sft_data.jsonl")
    args = parser.parse_args()
    samples = build()
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n_tool = sum(1 for s in samples
                 if any(m["role"] == "tool" for m in s["conversations"]))
    print(f"{args.out}: {len(samples)} samples, {n_tool} with tool calls")