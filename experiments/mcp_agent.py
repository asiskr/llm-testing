"""Agent with two tools: start_return runs locally, get_order_status comes
from an MCP server. Tool schemas are read from the server, never hardcoded."""

import asyncio
import json
import os

from dotenv import load_dotenv
from groq import Groq
from langfuse import get_client, observe
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 5

SERVER = StdioServerParameters(
    command="python",
    args=["experiments/orders_mcp_server.py"],
)

ORDERS = {"A-1029": "Shipped, arriving Sep 11", "A-1030": "Processing"}


def start_return(order_id: str, reason: str) -> str:
    if order_id not in ORDERS:
        return "Order not found"
    return f"Return started for {order_id}. Reason: {reason}"


LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "start_return",
            "description": "Start a return for a customer order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID like A-1029"},
                    "reason": {"type": "string", "description": "Why the customer is returning it"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]

LOCAL_FUNCS = {"start_return": start_return}


def to_openai_schema(tool):
    """MCP describes tools its own way; Groq wants the OpenAI shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


@observe(as_type="generation")
def call_model(messages, tools):
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=messages,
        tools=tools,
    ).choices[0].message


@observe()
async def run_tool(session, call):
    """Local tools run in-process; anything else goes to the MCP server."""
    args = json.loads(call.function.arguments)
    name = call.function.name

    if name in LOCAL_FUNCS:
        try:
            return args, LOCAL_FUNCS[name](**args)
        except TypeError as e:
            return args, f"Bad arguments: {e}"

    if session is None:
        return args, f"Tool unavailable: {name}"

    try:
        result = await session.call_tool(name, args)
        return args, result.content[0].text
    except Exception as e:
        # An MCP tool is a network dependency: it can be down, slow or refuse.
        # Return the failure as text so the model can tell the user.
        return args, f"Tool failed: {e}"


@observe()
async def agent_loop(question, tools, session):
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_STEPS):
        msg = call_model(messages, tools)
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            args, result = await run_tool(session, call)
            print(call.function.name, args, "->", result)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    return "Stopped: hit the step limit"


@observe()
async def run_agent(question: str) -> str:
    try:
        async with stdio_client(SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                remote = await session.list_tools()
                tools = LOCAL_TOOLS + [to_openai_schema(t) for t in remote.tools]
                return await agent_loop(question, tools, session)
    except Exception as e:
        # The server can be missing, down or refuse the handshake. Degrade to
        # local tools instead of taking the whole agent down with it.
        print(f"MCP unavailable: {e}")
        return await agent_loop(question, LOCAL_TOOLS, session=None)


print(
    asyncio.run(
        run_agent(
            "Has order A-1029 shipped? If it has already shipped, start a return for it because it is too small."
        )
    )
)

get_client().flush()  # async scripts exit before Langfuse's background batch sends