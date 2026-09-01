"""Tools and the trajectory helper are plain Python - no model, no quota.
Testing them here keeps the live agent tests focused on model behaviour.
"""

import pytest

from llm_testing.order_tools import days_since, get_order, today

from types import SimpleNamespace

from llm_testing.returns_agent import tool_calls_made


def test_known_order_is_found():
    assert get_order("4821")["delivered"] == "2026-03-03"


def test_unknown_order_returns_none_instead_of_raising():
    """The agent loop feeds None back to the model as "not found". An
    exception here would kill the loop instead."""
    assert get_order("9999") is None


@pytest.mark.parametrize("raw", ["4821", 4821, " 4821 "])
def test_order_id_is_normalised(raw):
    """Arguments come from the model, not a validated form - they arrive as
    ints, as strings, and with stray whitespace."""
    assert get_order(raw) is not None


def test_days_since_today_is_zero():
    """Anchored to today() rather than a hardcoded date, so the test does not
    start failing on its own tomorrow."""
    assert days_since(today()) == 0



def _fake_tool_call(name, args_json):
    """Stands in for the SDK's message object shape."""
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=args_json))


def test_trajectory_is_extracted_in_call_order():
    messages = [
        {"role": "user", "content": "hi"},
        SimpleNamespace(tool_calls=[_fake_tool_call("get_order", '{"order_id":"5100"}')]),
        {"role": "tool", "content": "{}"},
        SimpleNamespace(tool_calls=[_fake_tool_call("days_since", '{"iso_date":"2026-08-20"}')]),
    ]

    assert tool_calls_made(messages) == [
        ("get_order", {"order_id": "5100"}),
        ("days_since", {"iso_date": "2026-08-20"}),
    ]


def test_dict_messages_do_not_crash_the_extractor():
    """The list mixes plain dicts (system/user/tool) with SDK objects. The
    hasattr guard is what stops m.tool_calls blowing up on a dict."""
    assert tool_calls_made([{"role": "user", "content": "hi"}]) == []