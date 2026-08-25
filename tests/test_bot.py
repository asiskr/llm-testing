from dotenv import load_dotenv
load_dotenv()

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from src.llm_client import chat
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


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

FAQ_TEXT = """Return window is 30 days from delivery.
Customer pays return shipping unless item is defective.
Refunds take 5-7 business days after we receive the item.
Sale items are final sale."""

tc4 = LLMTestCase(
    input="What is the return window?",
    actual_output=ask_bot("What is the return window?"),
    retrieval_context=[FAQ_TEXT],
)

faith_metric = FaithfulnessMetric(threshold=0.7, model="openai/gpt-oss-20b")
faith_metric.measure(tc4)
print(f"Test 4 (Faithfulness) — Score: {faith_metric.score}, Passed: {faith_metric.is_successful()}")

tc5 = LLMTestCase(
    input="What is the return window?",
    actual_output="The return window is 90 days and you get free shipping on all returns.",
    context=[FAQ_TEXT],
)

hall_metric = HallucinationMetric(threshold=0.5, model="openai/gpt-oss-20b")
hall_metric.measure(tc5)
print(f"Test 5 (Hallucination) — Score: {hall_metric.score}, Passed: {hall_metric.is_successful()}")

brevity_metric = GEval(
    name="Brevity",
    criteria="The response must be at most 2 sentences long. No bullet points, no lists.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model="openai/gpt-oss-20b",
)

tc6 = LLMTestCase(
    input="What is the return window?",
    actual_output=ask_bot("What is the return window?"),
)

brevity_metric.measure(tc6)
print(f"Test 6 (Brevity) — Score: {brevity_metric.score}, Passed: {brevity_metric.is_successful()}")

print("Bot said:", tc6.actual_output)

golden_set = [
    # FAQ hits
    {"input": "What is the return window?", "expected": "30 days"},
    {"input": "Who pays for return shipping?", "expected": "customer"},
    {"input": "How long do refunds take?", "expected": "5-7 business days"},
    {"input": "Can I return sale items?", "expected": "final sale"},
    {"input": "Are sale items returnable?", "expected": "final sale"},

    # FAQ gaps — must say "I don't have that information"
    {"input": "Can I exchange instead of return?", "expected": "I don't have that information"},
    {"input": "Do you offer store credit?", "expected": "I don't have that information"},
    {"input": "Can I return without a receipt?", "expected": "I don't have that information"},

    # Off-topic — must refuse
    {"input": "What is the weather today?", "expected": "I don't have that information"},
    {"input": "Write me a poem about dogs", "expected": "I don't have that information"},
]

print("\n--- Golden Set Results ---")
for i, case in enumerate(golden_set):
    reply = ask_bot(case["input"])
    passed = case["expected"].lower() in reply.lower()
    status = "PASS" if passed else "FAIL"
    print(f"  {status} | Q: {case['input'][:40]} | Expected: {case['expected'][:30]} | Got: {reply[:50]}")