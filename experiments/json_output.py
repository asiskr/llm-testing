import json
import sys

sys.path.append(".")

from llm_testing.llm_client import ask

prompt_plain = "Extract name and age from: Rahul is 25 years old and lives in Pune."

print(ask(prompt_plain))


prompt_json = """
Extract the name and age from the following text.

Text: Rahul is 25 years old and lives in Pune.

Return the output as valid JSON.
Use exactly these keys: "name" and "age".
Do not include any explanation, markdown, or extra text.
"""

print(ask(prompt_json))

result = ask(prompt_json)
print(result[0])
print("----------------------------")
data = json.loads(result)
print(type(data))
print("----------------------------")
print(data["name"])
print("----------------------------")
print(type(data["age"]))
print("----------------------------")

assert "name" in data
assert type(data["age"]) is int
print("PASS")
