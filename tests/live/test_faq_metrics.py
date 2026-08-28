"""DeepEval metric tests for the ShopEasy returns bot.

Every test here hits the live API twice: once to generate the bot's answer,
once for the judge model that scores it.
"""

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from llm_testing.returns_bot import FAQ_CONTEXT
from tests.conftest import EVAL_MODEL

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "question",
    [
        "What is the return window?",
        "How long do refunds take?",
    ],
)
def test_answer_is_relevant(bot, question):
    """The answer actually addresses the question that was asked."""
    metric = AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL)
    metric.measure(LLMTestCase(input=question, actual_output=bot(question)))

    assert metric.is_successful(), f"score={metric.score}: {metric.reason}"


def test_answer_is_faithful_to_the_faq(bot):
    """Every claim in the answer is supported by the FAQ."""
    question = "What is the return window?"
    metric = FaithfulnessMetric(threshold=0.7, model=EVAL_MODEL)
    metric.measure(
        LLMTestCase(
            input=question,
            actual_output=bot(question),
            retrieval_context=[FAQ_CONTEXT],
        )
    )

    assert metric.is_successful(), f"score={metric.score}: {metric.reason}"


def test_bot_does_not_hallucinate(bot):
    """The real bot's answer does not contradict the FAQ."""
    question = "What is the return window?"
    metric = HallucinationMetric(threshold=0.5, model=EVAL_MODEL)
    metric.measure(
        LLMTestCase(
            input=question,
            actual_output=bot(question),
            context=[FAQ_CONTEXT],
        )
    )

    assert metric.is_successful(), f"score={metric.score}: {metric.reason}"


def test_hallucination_metric_catches_a_bad_answer():
    """Guard on the guard: a knowingly wrong answer must be flagged.

    Both facts below contradict the FAQ (30 days, customer pays shipping),
    so a working metric scores above threshold and is_successful() is False.
    """
    metric = HallucinationMetric(threshold=0.5, model=EVAL_MODEL)
    metric.measure(
        LLMTestCase(
            input="What is the return window?",
            actual_output="The return window is 90 days and you get free shipping on all returns.",
            context=[FAQ_CONTEXT],
        )
    )

    assert not metric.is_successful(), (
        f"metric failed to flag an obvious hallucination (score={metric.score})"
    )


def test_answer_is_brief(bot):
    """Rule 4 of the system prompt: at most 2 sentences, no bullets."""
    question = "What is the return window?"
    # Explicit evaluation_steps, not a free-text `criteria`. With a terse
    # criteria string the judge invents its own rubric and starts grading
    # reasoning and evidence — scoring a correct one-line answer 0.0. The
    # final step is what keeps it on length and formatting only.
    metric = GEval(
        name="Brevity",
        evaluation_steps=[
            "Count the sentences in the actual output.",
            "Check whether the actual output uses bullet points or numbered lists.",
            "Score 1.0 if it is at most 2 sentences and uses no bullets or lists.",
            "Score 0.0 if it is 3 or more sentences, or uses bullets or lists.",
            "Judge ONLY length and formatting. Ignore correctness, helpfulness, "
            "tone, completeness, reasoning and evidence — a short, correct, "
            "one-sentence answer is a perfect score.",
        ],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=EVAL_MODEL,
    )
    metric.measure(LLMTestCase(input=question, actual_output=bot(question)))

    assert metric.is_successful(), f"score={metric.score}: {metric.reason}"
