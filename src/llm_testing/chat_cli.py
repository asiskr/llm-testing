"""Interactive terminal chat with the ShopEasy returns bot.

Run from the project root:

    python -m src.chat_cli

Type 'exit' to quit. Every exchange is appended to chat_log.jsonl.
"""

import json

from llm_testing.llm_client import chat
from llm_testing.returns_bot import SYSTEM_PROMPT

LOG_FILE = "chat_log.jsonl"


def log(user_msg, bot_reply):
    entry = {"user": user_msg, "bot": bot_reply}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ").strip()
        if user_input == "exit":
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print("Bot:", reply)
        log(user_input, reply)


if __name__ == "__main__":
    main()
