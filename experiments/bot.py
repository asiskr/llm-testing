from src.llm_client import chat
import json

def log(user_msg, bot_reply):
    entry = {
        "user": user_msg,
        "bot": bot_reply,
    }
    with open("chat_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

history = [
    {"role": "system", "content": """You are a returns support agent for ShopEasy.
Rules:
1. Answer ONLY using the FAQ below.
2. Do NOT add any information that is not in the FAQ.
3. If the answer is not in the FAQ, say exactly: "I don't have that information."
4. Keep answers under 2 sentences. No bullet points, no extra detail.

<faq>
Q: What is the return window?
A: 30 days from delivery.

Q: Who pays return shipping?
A: The customer, unless the item is defective.

Q: How long do refunds take?
A: 5-7 business days after we receive the item.

Q: Can I return sale items?
A: No, all sale items are final sale.
</faq>
"""},
]

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