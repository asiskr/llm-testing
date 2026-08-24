from src.llm_client import ask

prompt = """Extract the age from this text.

Text: Rahul is 25 years old."""

response = ask(prompt, temperature=1)
print("GOT:", repr(response))

assert response == "25" 
print("PASS")