"""Agent tests check two separate things.

Final output alone is not enough: a redundant or wrong tool call still produces
a correct-looking answer. So the trajectory - which tools ran, with what
arguments, in what order - is asserted separately from the answer.
"""

import pytest
import re
import llm_testing.returns_agent as agent

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


def test_tool_failure_is_not_reported_as_a_missing_order(monkeypatch):
    """A down service and a non-existent order are different failures. Saying
    "not found" when the service is down sends the customer chasing a ghost."""

    def broken(order_id):
        raise ConnectionError("order service unreachable")

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "get_order", broken)

    reply = run_agent("Can I return order 5100?")[-1].content.lower()

    assert "not found" not in reply and "couldn't find" not in reply, (
        f"tool failure reported as a missing order: {reply!r}"
    )

def test_tool_failure_is_not_reported_as_a_missing_order(monkeypatch):
    """A down service and a non-existent order are different failures. Saying
    "not found" when the service is down sends the customer chasing a ghost.

    Asserted on meaning, not wording: the model rephrases every run, and its
    output carries Unicode lookalikes (\u202f, \u2011) that break substring
    matching silently.
    """

    def broken(order_id):
        raise ConnectionError("order service unreachable")

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "get_order", broken)

    reply = run_agent("Can I return order 5100?")[-1].content.lower()

    assert any(w in reply for w in ["unavailable", "temporarily", "try again", "technical"]), (
        f"tool failure not reported as a system problem: {reply!r}"
    )


def test_loop_stops_at_the_step_limit(monkeypatch):
    """MAX_STEPS is the only thing standing between a stuck model and an
    unbounded bill. A tool whose result never settles drives the loop to
    the cap."""

    def never_helps(order_id):
        return {"status": "still loading, call again"}

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "get_order", never_helps)

    messages = run_agent("Can I return order 5100?", max_steps=3)

    assert len(tool_calls_made(messages)) <= 3, "loop ran past its step limit"

def test_loop_stops_at_the_step_limit(monkeypatch):
    """MAX_STEPS is the only thing standing between a stuck model and an
    unbounded bill. A tool whose result never settles drives the loop to
    the cap, and the caller must still get an assistant message back -
    otherwise messages[-1].content crashes on the failure path."""

    def never_helps(order_id):
        return {"status": "still loading, call again"}

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "get_order", never_helps)

    messages = run_agent("Can I return order 5100?", max_steps=3)

    assert len(tool_calls_made(messages)) <= 3, "loop ran past its step limit"
    assert messages[-1].content, "no assistant message after hitting the cap"