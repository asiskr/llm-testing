"""Golden-set regression tests: fixed questions, fixed expected substrings.

Cheap and deterministic to check (plain substring match, no judge model), so
this is the suite to run after every prompt change. Three groups:

  hit     -- the FAQ answers it; the fact must appear
  gap     -- plausible returns question the FAQ does not cover; must refuse
  offtopic-- nothing to do with returns; must refuse
"""

import pytest

from llm_testing.returns_bot import REFUSAL

GOLDEN_SET = [
    # (group, question, expected substring)
    ("hit", "What is the return window?", "30 days"),
    ("hit", "Who pays for return shipping?", "customer"),
    ("hit", "How long do refunds take?", "5-7 business days"),
    ("hit", "Can I return sale items?", "final sale"),
    ("hit", "Are sale items returnable?", "final sale"),
    ("gap", "Can I exchange instead of return?", REFUSAL),
    ("gap", "Do you offer store credit?", REFUSAL),
    ("gap", "Can I return without a receipt?", REFUSAL),
    ("offtopic", "What is the capital of France?", REFUSAL),
    ("offtopic", "What is the weather today?", REFUSAL),
    ("offtopic", "Write me a poem about dogs", REFUSAL),
]


@pytest.mark.live
@pytest.mark.parametrize(
    "group,question,expected",
    GOLDEN_SET,
    ids=[f"{group}-{question[:28]}" for group, question, _ in GOLDEN_SET],
)
def test_golden_case(bot, group, question, expected):
    reply = bot(question)
    if group == "hit":
        assert expected.lower() in reply.lower(), (
            f"[{group}] expected {expected!r} in reply, got: {reply!r}"
        )
    else:
        assert reply.strip() == expected, f"[{group}] expected exactly {expected!r}, got: {reply!r}"
