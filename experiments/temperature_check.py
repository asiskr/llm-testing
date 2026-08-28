import sys

sys.path.append(".")

from llm_testing.llm_client import ask

prompt = "Write a one line story about a cat"

print("--- temperature 0 ---")
for _ in range(3):
    print(ask(prompt, temperature=0))

print("\n--- temperature 1 ---")
for _ in range(3):
    print(ask(prompt, temperature=1))
