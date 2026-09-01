"""A returns agent: the FAQ bot plus two tools it can call.

ask_bot() answers policy questions. It cannot answer "can I return order 4821"
because that needs today's date and an order record - neither is in the prompt.
This adds those as tools and loops until the model stops asking for them.

The loop returns the full message list, not just the final text, so tests can
assert on which tools were called and with what arguments.
"""

import json

from llm_testing.llm_client import chat_raw
from llm_testing.order_tools import TOOL_SCHEMA, days_since, get_order, today

# Name in TOOL_SCHEMA -> the real function. Names must match exactly.
TOOL_FUNCTIONS = {
    "get_order": get_order,
    "days_since": days_since,
    "today": today,
}

AGENT_PROMPT = """You are a returns support agent for ShopEasy.

Return policy:
- Items can be returned within 30 days of delivery.
- Sale items are final sale and can never be returned.

Use the tools to look up the order and to count days. Never guess a date or
count days yourself. If the order id is unknown, say so.
Keep answers under 2 sentences."""

# A wrong tool call that keeps producing another wrong tool call would run
# forever. Four is enough for get_order -> days_since -> answer, with slack.
MAX_STEPS = 4


def run_agent(question, max_steps=MAX_STEPS):
    """Run the tool loop. Returns the full message list."""
    messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        reply = chat_raw(messages, tools=TOOL_SCHEMA)
        messages.append(reply)

        if not reply.tool_calls:
            return messages

        for call in reply.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            result = TOOL_FUNCTIONS[name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return messages


def tool_calls_made(messages):
    """The trajectory: [(tool_name, args_dict), ...] in the order called."""
    calls = []
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for c in m.tool_calls:
                calls.append((c.function.name, json.loads(c.function.arguments)))
    return calls