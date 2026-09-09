from llm_testing import llm_client
from llm_testing.returns_agent import run_agent, tool_calls_made

llm_client.reset_token_count()
messages = run_agent("Can I return order 5100?")

print(len(tool_calls_made(messages)), "tool calls")
print(llm_client.TOKENS_USED, "tokens")