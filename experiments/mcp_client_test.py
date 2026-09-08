"""Smallest possible MCP client: connect to the server and list its tools."""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server = StdioServerParameters(
    command="python",
    args=["experiments/orders_mcp_server.py"],
)


async def main():
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            for t in tools.tools:
                print(t.name, "-", t.description)

            result = await session.call_tool(
                "get_order_status", {"order_id": "A-1029"}
            )
            print(result.content[0].text)
asyncio.run(main())