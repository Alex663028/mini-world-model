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

运行（stdio 模式，由 MCP 客户端拉起）：
    H:/ComfyUI_rocm/python_env/python.exe mcp_server.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from agent import MiniWorldAgent

mcp = FastMCP("mini-world-agent")

_bot = None


def bot():
    global _bot
    if _bot is None:
        _bot = MiniWorldAgent()
    return _bot


@mcp.tool()
def agent_reply(text: str) -> str:
    """统一入口：对 MiniWorldAgent 说一句话，自动路由到对应能力回答。

    Args:
        text: 要说的内容，如"去目标那儿"、"你看到了什么？"、"光速是多少？"、"23 乘以 7"
    """
    return bot().reply(text)


@mcp.tool()
def world_state() -> dict:
    """当前环境状态：agent 位置、目标位置、直线距离、步数。"""
    import numpy as np

    b = bot()
    return {
        "agent_pos": [round(float(v), 3) for v in b.env.pos],
        "target_pos": [round(float(v), 3) for v in b.env.target],
        "distance": round(float(np.linalg.norm(b.env.pos - b.env.target)), 3),
        "steps": b.env.steps,
    }


@mcp.tool()
def world_perceive() -> str:
    """感知：真跑世界模型编码器，用自然语言描述当前看到的画面。"""
    return bot().reply("你看到了什么？")


@mcp.tool()
def world_navigate(task: str = "去目标那儿") -> str:
    """任务：MPC 规划 + PD 控制导航到目标，带进度汇报（完整 agent 闭环）。

    Args:
        task: 任务描述，默认"去目标那儿"
    """
    return bot().reply(task)


@mcp.tool()
def world_imagine(direction: str = "前") -> str:
    """世界模型想象：不碰真实环境，在潜空间 rollout 12 步，报告梦里位置。

    Args:
        direction: 想象的方向：前/后/左/右
    """
    return bot().reply(f"想象一下往{direction}走")


@mcp.tool()
def world_act(direction: str = "前") -> str:
    """在真实环境执行一个动作。

    Args:
        direction: 方向：前/后/左/右
    """
    return bot().reply(f"往{direction}走")


@mcp.tool()
def world_reset() -> str:
    """重置环境：agent 和目标到新随机位置，开新回合。"""
    return bot().reply("重置")


@mcp.tool()
def kb_query(question: str) -> str:
    """常识库检索（bigram 匹配，40+ 条原创常识）。

    Args:
        question: 要查的问题，如"光速是多少"
    """
    hit = bot()._retrieve(question)
    return hit[0] if hit else "常识库没有这条，试试 agent_reply 让语言模型回答"


@mcp.tool()
def calc(expression: str) -> str:
    """符号计算器：四则运算（语言模型不碰算术，工具比肌肉可靠）。

    Args:
        expression: 算式，如"23 乘以 7"、"100 除以 4"
    """
    return bot()._try_calc(expression) or "无法解析这个算式"


if __name__ == "__main__":
    mcp.run()  # stdio 传输，由 MCP 客户端拉起
