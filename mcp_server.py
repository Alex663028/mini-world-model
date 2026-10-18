"""MiniWorldAgent MCP Server：把整个 agent 的能力暴露成标准 MCP 工具。

任何支持 MCP 的客户端（WorkBuddy、Claude Desktop 等）连上后，
即可调用小 WM 的全部能力：感知世界模型内部状态、下导航任务、
世界模型想象、常识查询、符号计算。

工具清单（9 个）：
    agent_reply     统一入口：自然语言 → 系统路由回答
    world_state     环境状态（位置/目标/距离）
    world_perceive  感知：潜状态 + 文字描述（真跑世界模型）
    world_navigate  任务：MPC+PD 导航到目标，带进度汇报
    world_imagine   世界模型想象 rollout
    world_act       单步动作（方向）
    world_reset     重置环境
    kb_query        常识库检索
    calc            符号计算器

容错约定：每个工具都自己兜异常，失败时返回可读的错误文本（dict 工具返回
{"ok": false, "error": ...}）并往 stderr 打 traceback——工具内部报错不该把
MCP 会话打断，也不该让客户端只看到一句 "internal error"。

运行（stdio 模式，由 MCP 客户端拉起）：
    pip install mcp        # 依赖未装会直接给出这句提示
    python mcp_server.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # 依赖缺失时给人话，而不是 ImportError traceback
    raise SystemExit(
        "mcp_server.py 需要 mcp 包：\n"
        "    pip install mcp          # 或 pip install -r requirements.txt\n"
        f"原始错误：{e}"
    )

from agent import MiniWorldAgent

mcp = FastMCP("mini-world-agent")

_bot = None


def bot():
    global _bot
    if _bot is None:
        try:
            _bot = MiniWorldAgent()
        except SystemExit as e:
            # 权重缺失等致命配置问题：agent 会 SystemExit，MCP 侧转成可读错误，
            # 别让工具调用直接把 server 进程带走
            raise RuntimeError(
                f"MiniWorldAgent 初始化失败（exit={e}）；"
                f"请先准备 checkpoint.pt / 语言模型权重（见 README 路线 A/B）"
            ) from e
    return _bot


def _fail(tool, e):
    """工具失败的统一出口：stderr 留痕 + 返回可读字符串。"""
    sys.stderr.write(f"[mcp] {tool} 失败: {type(e).__name__}: {e}\n")
    sys.stderr.write(traceback.format_exc())
    return f"{tool} 执行失败：{type(e).__name__}: {e}"


@mcp.tool()
def agent_reply(text: str) -> str:
    """统一入口：对 MiniWorldAgent 说一句话，自动路由到对应能力回答。

    Args:
        text: 要说的内容，如"去目标那儿"、"你看到了什么？"、"光速是多少？"、"23 乘以 7"
    """
    try:
        return bot().reply(text)
    except Exception as e:
        return _fail("agent_reply", e)


@mcp.tool()
def world_state() -> dict:
    """当前环境状态：agent 位置、目标位置、直线距离、步数。

    失败时返回 {"ok": false, "error": "..."}。
    """
    import numpy as np

    try:
        b = bot()
        return {
            "ok": True,
            "agent_pos": [round(float(v), 3) for v in b.env.pos],
            "target_pos": [round(float(v), 3) for v in b.env.target],
            "distance": round(float(np.linalg.norm(b.env.pos - b.env.target)), 3),
            "steps": b.env.steps,
        }
    except Exception as e:
        sys.stderr.write(f"[mcp] world_state 失败: {type(e).__name__}: {e}\n")
        sys.stderr.write(traceback.format_exc())
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def world_perceive() -> str:
    """感知：真跑世界模型编码器，用自然语言描述当前看到的画面。"""
    try:
        return bot().reply("你看到了什么？")
    except Exception as e:
        return _fail("world_perceive", e)


@mcp.tool()
def world_navigate(task: str = "去目标那儿") -> str:
    """任务：MPC 规划 + PD 控制导航到目标，带进度汇报（完整 agent 闭环）。

    注意：一次调用可能要跑 30 秒以上（逐步想象规划），客户端超时设置别太短。

    Args:
        task: 任务描述，默认"去目标那儿"
    """
    try:
        return bot().reply(task)
    except Exception as e:
        return _fail("world_navigate", e)


@mcp.tool()
def world_imagine(direction: str = "前") -> str:
    """世界模型想象：不碰真实环境，在潜空间 rollout 12 步，报告梦里位置。

    Args:
        direction: 想象的方向：前/后/左/右
    """
    try:
        return bot().reply(f"想象一下往{direction}走")
    except Exception as e:
        return _fail("world_imagine", e)


@mcp.tool()
def world_act(direction: str = "前") -> str:
    """在真实环境执行一个动作。

    Args:
        direction: 方向：前/后/左/右
    """
    try:
        return bot().reply(f"往{direction}走")
    except Exception as e:
        return _fail("world_act", e)


@mcp.tool()
def world_reset() -> str:
    """重置环境：agent 和目标到新随机位置，开新回合。"""
    try:
        return bot().reply("重置")
    except Exception as e:
        return _fail("world_reset", e)


@mcp.tool()
def kb_query(question: str) -> str:
    """常识库检索（bigram 匹配；小库 + 大库全量索引，见 knowledge.py）。

    Args:
        question: 要查的问题，如"光速是多少"
    """
    try:
        hit = bot()._retrieve(question)
        return hit[0] if hit else "常识库没有这条，试试 agent_reply 让语言模型回答"
    except Exception as e:
        return _fail("kb_query", e)


@mcp.tool()
def calc(expression: str) -> str:
    """符号计算器：四则运算（语言模型不碰算术，工具比肌肉可靠）。

    Args:
        expression: 算式，如"23 乘以 7"、"100 除以 4"
    """
    try:
        return bot()._try_calc(expression) or "无法解析这个算式"
    except Exception as e:
        return _fail("calc", e)


if __name__ == "__main__":
    mcp.run()  # stdio 传输，由 MCP 客户端拉起
