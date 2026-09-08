"""Tool-calling agent loop: the model picks tools, we run them and feed the
results back until it answers in words."""

import json
import os

from dotenv import load_dotenv
from langfuse import observe
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 5  # stops a tool-loop from burning quota forever

ORDERS = {
    "A-1029": "Shipped, arriving Sep 11",
    "A-1030": "Processing",
}


def get_order_status(order_id: str) -> str:
    return ORDERS.get(order_id, "Order not found")


def start_return(order_id: str, reason: str) -> str:
    if order_id not in ORDERS:
        return "Order not found"
    return f"Return started for {order_id}. Reason: {reason}"


# The description is the only thing the model reads to decide, so it is
# prompt text, not documentation.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the delivery status of a customer order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID like A-1029"},
                },
                "required": ["order_id"],
            },
        },
    },
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

TOOL_FUNCS = {
    "get_order_status": get_order_status,
    "start_return": start_return,
}
@observe()
def run_tool(call):
    """Run one tool call. Model errors come back as text, never as a crash."""
    args = json.loads(call.function.arguments)
    fn = TOOL_FUNCS.get(call.function.name)
    if fn is None:
        return args, f"Unknown tool: {call.function.name}"
    try:
        return args, fn(**args)
    except TypeError as e:
        return args, f"Bad arguments: {e}"

@observe(as_type="generation")
def call_model(messages):
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=messages,
        tools=TOOLS,
    ).choices[0].message

@observe()
def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_STEPS):
        msg = call_model(messages)

        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            args, result = run_tool(call)
            print(call.function.name, args, "->", result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Stopped: hit the step limit"


print(run_agent("Has order A-1029 shipped? If it has already shipped, start a return for it because it is too small."))