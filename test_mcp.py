"""MCP server 端到端测试：拉起 stdio 客户端调用全部工具。"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PYTHON = "H:/ComfyUI_rocm/python_env/python.exe"
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


async def main():
    params = StdioServerParameters(command=PYTHON, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"tools: {len(tools.tools)}")

            for name, args in [
                ("world_state", {}),
                ("kb_query", {"question": "光速是多少"}),
                ("calc", {"expression": "23 乘以 7"}),
                ("world_perceive", {}),
                ("world_act", {"direction": "左"}),
                ("world_reset", {}),
                ("agent_reply", {"text": "墙在哪？"}),
            ]:
                result = await session.call_tool(name, args)
                text = result.content[0].text if result.content else ""
                print(f"\n[{name}] {args}")
                print(f"  -> {text[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
