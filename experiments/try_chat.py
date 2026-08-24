from src.llm_client import chat

msgs = [
    {"role": "system", "content": "You are a helpful assistant. Keep answers short."},
    {"role": "user",   "content": "What is my name?"},
]

print(chat(msgs))