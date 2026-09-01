"""Thin wrapper around the Groq chat-completions API.

The client is built lazily on first use rather than at import time, so that
importing this module needs no credentials. That is what lets the offline
tests (and CI) collect and run with no GROQ_API_KEY set at all.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

_client = None


def get_client():
    """Return the shared Groq client, constructing it on first call."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def ask(prompt, temperature=0):
    """Send a single user prompt and return the reply text."""
    return chat([{"role": "user", "content": prompt}], temperature)


def chat(messages, temperature=0):
    """Send a full message list (system/user/assistant) and return the reply text."""
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def chat_raw(messages, tools=None, temperature=0):
    """Return the full message object, not just its text.

    Tool calls arrive in message.tool_calls and message.content is None in
    that case, so an agent loop cannot use chat() - it would silently get
    None back. Kept separate from chat() so existing callers are untouched.
    """
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    response = get_client().chat.completions.create(**kwargs)
    return response.choices[0].message