"""Agent 层测试：工具调用解析、方位计算、权重缺失的优雅失败。

不依赖任何权重文件——所有断言都在纯函数或错误路径上。
需要权重的端到端行为由 verify_state_bpe.py / eval_*.py 覆盖。
"""

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import agent as agent_mod  # noqa: E402
from agent import MiniWorldAgent  # noqa: E402


def test_parse_tool_plain():
    name, kwargs, rest = MiniWorldAgent._parse_tool("[调用工具 world_navigate]")
    assert name == "world_navigate"
    assert kwargs == {}


def test_parse_tool_with_params():
    name, kwargs, _ = MiniWorldAgent._parse_tool(
        "[调用工具 world_act, direction=西北, magnitude=0.7]")
    assert name == "world_act"
    assert kwargs["direction"] == "西北"
    assert abs(kwargs["magnitude"] - 0.7) < 1e-6


def test_parse_tool_no_call():
    name, kwargs, rest = MiniWorldAgent._parse_tool("直接回答，不调工具")
    assert name is None


def test_bearing_labels():
    """方位命名必须覆盖 8 个方向。

    坐标约定：env._to_pixel 不做 y 翻转（世界 y 正方向 = 画面下方 = 南），
    所以 _bearing 里用 atan2(-dy, dx) 把世界系转成"上北下南"的直觉方位。
    这个约定如果改了，MPC 的方位汇报和事件记忆会全体错位——用测试钉死。
    """
    bot = MiniWorldAgent.__new__(MiniWorldAgent)
    origin = np.zeros(2, dtype=np.float32)
    cases = {
        (1.0, 0.0): "正东",
        (-1.0, 0.0): "正西",
        (0.0, 1.0): "正南",   # 世界 y 正 = 画面下 = 南
        (0.0, -1.0): "正北",
    }
    for vec, expect in cases.items():
        got = bot._bearing(origin, np.array(vec, dtype=np.float32))
        assert got == expect, f"{vec} 应为 {expect}，实际 {got}"


def test_missing_world_ckpt_exits_cleanly(tmp_path):
    """权重缺失时必须 SystemExit + 明确指引，不能抛 traceback。

    这是开源体验的底线：新用户 clone 后直接跑 agent.py 应该看到
    "先跑 python train.py"，而不是一堆 FileNotFoundError。
    """
    code = (
        f"import sys; sys.path.insert(0, {ROOT!r})\n"
        "from agent import MiniWorldAgent\n"
        "MiniWorldAgent(world_ckpt='definitely_not_here.pt')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, "缺权重应非零退出"
    assert "找不到世界模型权重" in proc.stderr, f"缺少友好提示：{proc.stderr[-300:]}"
    assert "train.py" in proc.stderr, "提示里应给出训练命令"


def test_config_defaults():
    """关键默认值：状态头默认关闭（实测导航 5/8 vs 7/8）、continuous 候选。"""
    cfg = agent_mod.CONFIG
    assert cfg["mpc_candidates"] == "continuous"
    assert cfg["use_state_head"] is False
    assert cfg["imagine_horizon"] == 12


def test_perception_keys_cover_common_phrasings():
    """感知问法的关键词覆盖（'往哪边'曾漏掉，被能力介绍劫持）。"""
    keys = agent_mod.PERCEPTION_KEYS
    for phrase in ["你看到了什么", "目标离你多远", "你现在在往哪边走", "目标在哪个方向"]:
        assert any(k in phrase for k in keys), f"缺少覆盖：{phrase}"


def test_task_and_reset_keys():
    for phrase in ["去目标那儿", "导航过去", "走到目标"]:
        assert any(k in phrase for k in agent_mod.TASK_KEYS), phrase
    for phrase in ["重置", "换一个目标", "新回合"]:
        assert any(k in phrase for k in agent_mod.RESET_KEYS), phrase
