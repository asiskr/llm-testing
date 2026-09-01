"""Agent tests check two separate things.

Final output alone is not enough: a redundant or wrong tool call still produces
a correct-looking answer. So the trajectory - which tools ran, with what
arguments, in what order - is asserted separately from the answer.
"""

import pytest
import re

from llm_testing.returns_agent import run_agent, tool_calls_made

pytestmark = pytest.mark.live


def test_looks_up_the_order_before_counting_days():
    """Trajectory: the delivery date has to come from the tool, not the model."""
    calls = tool_calls_made(run_agent("Can I return order 5100?"))
    names = [name for name, _ in calls]

    assert names[:2] == ["get_order", "days_since"], f"unexpected trajectory: {names}"


def test_passes_the_id_the_customer_asked_about():
    """Tool-call correctness: right tool is not enough, the args must be right."""
    calls = tool_calls_made(run_agent("Can I return order 5100?"))
    first_name, first_args = calls[0]

    assert first_name == "get_order"
    assert first_args["order_id"] == "5100", f"wrong id passed: {first_args}"


def test_does_not_produce_a_date_without_calling_a_tool():
    """The agent has no clock. A date in the reply that came from no tool call
    is a guess, even when it happens to be right."""
    messages = run_agent("What day is it today?")
    calls = tool_calls_made(messages)
    reply = messages[-1].content

    if not calls:
        assert not re.search(r"\d{4}[-\u2011]\d{2}[-\u2011]\d{2}", reply), (
            f"agent stated a date with no tool call: {reply!r}"
        )