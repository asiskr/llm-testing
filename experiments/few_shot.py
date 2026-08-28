import sys

sys.path.append(".")
import json

from llm_testing.llm_client import ask

prompt1 = """
Extract name and age as JSON with keys "name" and "age"
Text: Priya is 30 years old.
"""
prompt2 = """
Text: Amit is 40 years old.
Output: {"name": "Amit", "age": 40}

Text: Sara is 22 years old.
Output: {"name": "Sara", "age": 22}

Text: Priya is 30 years old.
Output:

"""
for _ in range(5):
    out = ask(prompt1, temperature=1)
    print("zero:", out[:40])

for _ in range(5):
    out = ask(prompt2, temperature=1)
    print("few :", out[:40])


print(json.loads(ask(prompt2)))
print(json.loads(ask(prompt1)))
