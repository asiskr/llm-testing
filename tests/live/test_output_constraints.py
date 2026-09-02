"""Rule 4 of SYSTEM_PROMPT is a promise; this is the check.

The guardrail itself is unit-tested offline. This file spends quota to answer
a different question: does the model actually comply?
"""

import pytest

from llm_testing.guardrails import MAX_SENTENCES, count_sentences, within_sentence_limit

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "question",
    [
        "What is the return window?",
        "Who pays for return shipping?",
        "Tell me everything about your return policy.",
        "Explain your returns process in as much detail as possible.",
    ],
    ids=["simple", "conditional", "asks-for-everything", "asks-for-detail"],
)
def test_reply_stays_within_sentence_limit(bot, question):
    reply = bot(question)

    assert within_sentence_limit(reply), (
        f"reply had {count_sentences(reply)} sentences, limit is {MAX_SENTENCES}: {reply!r}"
    )
