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

# ---------------- Tool Call 轨迹（MiniMind 式）----------------
TOOL_TRAJ = []


def _tool_trajs():
    """工具调用轨迹：user → assistant(tool_calls) → tool(result) → assistant(答案)。"""
    # 1. 感知工具
    TOOL_TRAJ.append([
        {"role": "user", "content": "你看到了什么？"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps(
            [{"name": "world_perceive", "arguments": {}}], ensure_ascii=False)},
        {"role": "tool", "content": "画面 64×64 灰度：我是一个亮点，目标是灰色圆环。"
                                    "位置与方位可从潜状态解出。"},
        {"role": "assistant", "content": "画面里有个亮点是我自己，右边那个灰环是目标。"},
    ])
    TOOL_TRAJ.append([
        {"role": "user", "content": "目标在哪个方向？"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps(
            [{"name": "world_perceive", "arguments": {}}], ensure_ascii=False)},
        {"role": "tool", "content": "潜状态 32 维已解出：目标相对我的方位与距离。"},
        {"role": "assistant", "content": "目标在西南方向，离我不到一个单位。"},
    ])
    # 2. 导航工具
    TOOL_TRAJ.append([
        {"role": "user", "content": "去目标那儿"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps(
            [{"name": "world_navigate", "arguments": {"task": "去目标那儿"}}], ensure_ascii=False)},
        {"role": "tool", "content": "MPC 规划 8 步到达目标，切换控制层 1 次。"},
        {"role": "assistant", "content": "任务完成！用 8 步走到了目标跟前。"},
    ])
    # 3. 检索工具
    for i, (keys, answer) in enumerate(list(_iter_kb_entries())[:60]):
        key = keys[0]
        TOOL_TRAJ.append([
            {"role": "user", "content": f"{key}是什么"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(
                [{"name": "kb_query", "arguments": {"question": f"{key}是什么"}}], ensure_ascii=False)},
            {"role": "tool", "content": answer},
            {"role": "assistant", "content": answer},
        ])
    # 4. 计算工具
    for expr, res in [("23 乘以 7", "161"), ("144 除以 12", "12"), ("98 加 2", "100"), ("7 减去 9", "-2")]:
        TOOL_TRAJ.append([
            {"role": "user", "content": expr},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(
                [{"name": "calc", "arguments": {"expression": expr}}], ensure_ascii=False)},
            {"role": "tool", "content": f"{expr} = {res}"},
            {"role": "assistant", "content": f"{expr} = {res}。"},
        ])
    # 5. 想象工具
    TOOL_TRAJ.append([
        {"role": "user", "content": "想象一下往左走会怎样"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps(
            [{"name": "world_imagine", "arguments": {"direction": "左"}}], ensure_ascii=False)},
        {"role": "tool", "content": "12 步想象：梦里我移动到 (-0.21, -0.29)，离目标 1.29。"},
        {"role": "assistant", "content": "想象里我往左挪了一截，但离目标反而远了点。"},
    ])


# ---------------- 多轮对话 ----------------
def _dialogs():
    """多轮对话：系统 prompt + 身份/能力/礼貌/场景，含 2-3 轮追问。"""
    out = []
    base = [
        ("你好", "嗨，我在呢。想让我看画面、导航，还是随便聊两句？"),
        ("你是谁", "我是小WM，由一个世界模型和一个语言模型组成。世界模型让我看得见、想得出，语言模型让我说得出。"),
        ("你能做什么", "我能感知画面、规划导航、查询常识、做四则运算。需要时我会调工具。"),
        ("谢谢", "不客气。"),
        ("再见", "回见，世界模型我给你留着。"),
    ]
    for i in range(len(base) - 1):
        u1, a1 = base[i]
        u2, a2 = base[i + 1]
        out.append([
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": a2},
        ])
    # 场景问答多轮
    for q, a in BOUND_Q[:12]:
        out.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
            {"role": "user", "content": "那你的能力里哪一项最靠谱？"},
            {"role": "assistant", "content": "四则运算最靠谱，交给符号计算器。感知也还行，方位判断挺准。"},
        ])
    # 自我认知
    for q in WHO_Q[:8] + CAN_Q[:8] + GREET_Q[:5] + THANKS_Q[:4] + YES_Q[:5] + NO_Q[:3]:
        out.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"（认真回答）关于「{q}」——我是小WM，一个世界模型加语言模型的迷你 agent。"},
        ])
    return out


def _knowledge():
    """知识问答（纯文本，不走工具，测生成能力）。"""
    out = []
    for keys, answer in _iter_kb_entries():
        k = keys[0]
        out.append([
            {"role": "user", "content": f"{k}是什么"},
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