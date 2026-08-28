"""Shared test fixtures.

Loads .env before anything imports the clients, and exposes a session-cached
bot so that asking the same question in two tests costs one API call.
"""

import functools

import pytest
from dotenv import load_dotenv

load_dotenv()

from llm_testing.returns_bot import ask_bot as _ask_bot  # noqa: E402  (must follow load_dotenv)

# Model used by DeepEval to *judge* the answers. Routed to Groq via the
# OPENAI_BASE_URL / OPENAI_API_KEY pair in .env — see README.
EVAL_MODEL = "openai/gpt-oss-20b"


@pytest.fixture(scope="session")
def bot():
    """Call the returns bot, memoised for the whole session."""

    @functools.cache
    def _cached(question):
        return _ask_bot(question)

    return _cached
