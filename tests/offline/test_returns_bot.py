"""Offline tests — zero API calls, zero quota, no credentials needed.

    pytest -m "not live"

Two kinds of check live here:

1. Prompt invariants. Cheap string assertions on the system prompt that catch
   the drift the live suite would only report as a confusing metric failure.
2. Message-assembly logic. ask_bot() builds the message list; that is ordinary
   code and deserves ordinary unit tests.

The mock patches "llm_testing.returns_bot.chat", NOT "llm_testing.llm_client.chat".
returns_bot did `from llm_testing.llm_client import chat`, so it holds its own
reference from import time and patching the defining module is a no-op — the
real API gets called and the test passes for the wrong reason. Patch where the
function is used, not where it is defined.
"""

import pytest

from llm_testing.returns_bot import FAQ_CONTEXT, REFUSAL, SYSTEM_PROMPT, ask_bot

# --- prompt invariants ----------------------------------------------------


def test_refusal_string_is_not_empty():
    assert REFUSAL.strip()


def test_system_prompt_contains_faq_tags():
    assert "<faq>" in SYSTEM_PROMPT
    assert "</faq>" in SYSTEM_PROMPT


def test_system_prompt_contains_the_exact_refusal_string():
    """The golden set asserts on REFUSAL. If the constant and the prompt drift
    apart, every gap and off-topic case fails for a reason that has nothing to
    do with the model."""
    assert REFUSAL in SYSTEM_PROMPT


@pytest.mark.parametrize("fact", ["30 days", "5-7 business days", "final sale"])
def test_faq_context_matches_the_prompt(fact):
    """FAQ_CONTEXT is the ground truth for the faithfulness and hallucination
    metrics. If it drifts from the prompt, those metrics score the bot against
    facts it was never given."""
    assert fact in SYSTEM_PROMPT
    assert fact in FAQ_CONTEXT


# --- message assembly, mocked --------------------------------------------


@pytest.fixture
def sent_messages(monkeypatch):
    """Replace chat() and capture what ask_bot() would have sent."""
    captured = []

    def _fake_chat(messages, temperature=0):
        captured.append(messages)
        return "30 days from delivery."

    monkeypatch.setattr("llm_testing.returns_bot.chat", _fake_chat)
    return captured


def test_ask_bot_returns_the_reply(sent_messages):
    assert "30 days" in ask_bot("What is the return window?")
    assert len(sent_messages) == 1, "expected exactly one (mocked) call"


def test_ask_bot_puts_system_prompt_first_and_question_last(sent_messages):
    ask_bot("What is the return window?")
    messages = sent_messages[0]

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1] == {"role": "user", "content": "What is the return window?"}


def test_ask_bot_threads_history_between_system_and_question(sent_messages):
    history = [
        {"role": "user", "content": "Can I return sale items?"},
        {"role": "assistant", "content": "No, all sale items are final sale."},
    ]
    ask_bot("Why not?", history=history)

    assert [m["role"] for m in sent_messages[0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_ask_bot_does_not_mutate_the_history_it_is_given(sent_messages):
    """A caller reusing its history list must not find the bot's scratch turns
    appended to it."""
    history = [{"role": "user", "content": "Can I return sale items?"}]
    ask_bot("Why not?", history=history)

    assert history == [{"role": "user", "content": "Can I return sale items?"}]
