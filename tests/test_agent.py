"""Agent 层测试：工具调用解析、方位计算、事件记忆、检索、权重缺失的优雅失败。

不依赖任何权重文件——所有断言都在纯函数、SimpleNamespace 假对象或错误路径上。
需要权重的端到端行为由 verify_state_bpe.py / eval_*.py 覆盖。
"""

import os
import re
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import agent as agent_mod  # noqa: E402
from agent import MiniWorldAgent, WorldState  # noqa: E402
from chat import bigrams  # noqa: E402
from corpus_dialog import WHO, WHO_Q  # noqa: E402
from env import PointMassEnv  # noqa: E402


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
    # 显式 utf-8 + errors=replace：子进程的中文提示在 Windows GBK 控制台下
    # 会解码失败（text=True 用 locale 编码），否则测试结果取决于环境变量
    proc = subprocess.run([sys.executable, "-c", code],
                          cwd=str(tmp_path), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                          timeout=120)
    assert proc.returncode != 0, "缺权重应非零退出"
    assert "找不到世界模型权重" in (proc.stderr or ""), f"缺少友好提示：{(proc.stderr or '')[-300:]}"
    assert "train.py" in (proc.stderr or ""), "提示里应给出训练命令"


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


# ---------- 死参数回归（H2：aim_horizon 写进黑洞）----------


def test_no_dead_aim_horizon_param():
    """回归：aim_horizon 全仓无消费方，必须保持删除状态。

    它曾让 eval_planner --horizons 的实验变成"horizon=12 重复测一遍"，
    于是 README 的"想象窗口已饱和"结论无有效实验支撑。
    """
    assert "aim_horizon" not in agent_mod.CONFIG, "CONFIG 里又出现死参数 aim_horizon"
    src = open(os.path.join(ROOT, "eval_planner.py"), encoding="utf-8").read()
    # 只看代码行：注释里出现 aim_horizon 是在解释历史，属正常
    code_lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert not re.search(r'\[["\']aim_horizon["\']\]', code), "eval_planner 又写回死参数了"
    assert re.search(r'\[["\']imagine_horizon["\']\]\s*=', code), \
        "eval_planner 必须写 imagine_horizon"


# ---------- 事件记忆：_step_env ----------


def test_step_env_records_wall_and_arrival():
    bot = MiniWorldAgent.__new__(MiniWorldAgent)
    bot.env = PointMassEnv(seed=11)
    bot.env.reset()
    bot.state = WorldState()
    bot.env.pos = np.array([0.99, 0.0], dtype=np.float32)
    bot.env.vel = np.zeros(2, dtype=np.float32)
    for _ in range(3):
        bot._step_env(np.array([1.0, 0.0], dtype=np.float32))
    assert bot.state.wall_hits >= 1, "朝墙推必须记录撞墙事件"
    assert bot.state.total_steps == 3

    # 到达：目标与当前位置重合
    bot.env = PointMassEnv(seed=12)
    bot.env.reset()
    bot.env.target = bot.env.pos.copy()
    bot.state = WorldState()
    _, _, done, info = bot._step_env(np.zeros(2, dtype=np.float32))
    assert done and info["reached"]
    assert bot.state.arrivals == 1, "到达必须记录 arrive 事件"


# ---------- 系统查表：_system_answer ----------


def test_system_answer_hits_table_and_is_stable():
    bot = MiniWorldAgent.__new__(MiniWorldAgent)
    q = WHO_Q[0]
    ans = bot._system_answer(q)
    assert ans in WHO, f"'{q}' 应命中 WHO 表，实际 {ans!r}"
    # 变体选择是确定函数（len(text) % len(a_list)），不能每次随机
    assert bot._system_answer(q) == WHO[len(q) % len(WHO)]
    assert bot._system_answer("希尔伯特空间的量子纠缠谱") is None


# ---------- 常识检索：_retrieve ----------


def test_retrieve_hit_and_miss():
    entry = {"keys": ["长城"], "answer": "长城是古代防御工程体系。"}
    obj = types.SimpleNamespace(cfg={"kb_threshold": 0.45},
                                kb=[(bigrams("长城"), entry)])
    hit = MiniWorldAgent._retrieve(obj, "长城是什么")
    assert hit is not None and hit[0] == entry["answer"]
    assert hit[1] >= 0.45, f"命中分数 {hit[1]} 应不低于阈值"
    assert MiniWorldAgent._retrieve(obj, "量子纠缠") is None, "无关问题不能误命中"
    assert MiniWorldAgent._retrieve(obj, "") is None, "空查询直接返回 None"
