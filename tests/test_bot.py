from dotenv import load_dotenv
load_dotenv()

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from src.llm_client import chat

SYSTEM = """You are a returns support agent for ShopEasy.
Rules:
1. Answer ONLY using the FAQ below.
2. Do NOT add any information not in the FAQ.
3. If the answer is not in the FAQ, say exactly: "I don't have that information."
4. Keep answers under 2 sentences. No bullet points.

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
"""

def ask_bot(question):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    return chat(msgs)

metric = AnswerRelevancyMetric(threshold=0.7, model="openai/gpt-oss-20b")

# Test 1: FAQ question
tc1 = LLMTestCase(input="What is the return window?", actual_output=ask_bot("What is the return window?"))
metric.measure(tc1)
print(f"Test 1 — Score: {metric.score}, Passed: {metric.is_successful()}")

# Test 2: Another FAQ question
tc2 = LLMTestCase(input="How long do refunds take?", actual_output=ask_bot("How long do refunds take?"))
metric.measure(tc2)
print(f"Test 2 — Score: {metric.score}, Passed: {metric.is_successful()}")

tc3_reply = ask_bot("What is the capital of France?")
assert "I don't have that information" in tc3_reply, f"Expected refusal, got: {tc3_reply}"
print("Test 3 — Passed: True (refusal check)")