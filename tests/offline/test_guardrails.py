"""The guardrail itself is deterministic, so it is tested offline and free.
Whether the bot's replies pass it is a separate, live test.
"""

import pytest

from llm_testing.guardrails import count_sentences, within_sentence_limit


@pytest.mark.parametrize(
    "text,expected",
    [
        ("30 days from delivery.", 1),
        ("The customer pays. Unless defective.", 2),
        ("One. Two. Three.", 3),
        ("Refunds take 5-7 business days.", 1),
        ("Version v2.0 applies.", 1),
        # Known limit: abbreviations over-count. Pinned so a future change
        # to the regex shows up here instead of surprising someone later.
        ("Contact Mr. Sharma today.", 2),
    ],
)
def test_sentence_count(text, expected):
    assert count_sentences(text) == expected


def test_limit_rejects_three_sentences():
    assert not within_sentence_limit("One. Two. Three.")
