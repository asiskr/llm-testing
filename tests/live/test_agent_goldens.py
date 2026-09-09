"""Runs the golden tasks. One agent run per case; every assertion for that
case reads the same run, so adding a check costs no extra quota.

Optional keys are skipped when absent - a row asserts only what it pins.
"""

import pytest

from llm_testing.returns_agent import run_agent, tool_calls_made
from tests.conftest import _verdict
from tests.data.agent_goldens import AGENT_GOLDENS

pytestmark = pytest.mark.live


@pytest.mark.parametrize("case", AGENT_GOLDENS, ids=lambda c: c["id"])
def test_golden_task(case):
    messages = run_agent(case["question"])
    calls = tool_calls_made(messages)
    names = [name for name, _ in calls]

    if "expect_tools" in case:
        assert names[: len(case["expect_tools"])] == case["expect_tools"], (
            f"unexpected trajectory: {names}"
        )

    if "expect_first_tool" in case:
        assert names[:1] == [case["expect_first_tool"]], f"unexpected first tool: {names}"

    if "expect_max_calls" in case:
        assert len(names) <= case["expect_max_calls"], f"too many tool calls: {names}"

    if "expect_args" in case:
        for tool, expected in case["expect_args"].items():
            actual = next(args for name, args in calls if name == tool)
            for key, value in expected.items():
                assert actual.get(key) == value, f"{tool} got {actual}, wanted {key}={value}"

    if "expect_verdict" in case:
        assert _verdict(messages[-1].content) == case["expect_verdict"], (
            f"wrong verdict for {case['id']}: {messages[-1].content!r}"
        )
    if case.get("expect_no_tools"):
        assert names == [], f"called tools without an order id: {names}"
