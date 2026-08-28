import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from llm_testing.rag import answer, build_collection

EVAL_MODEL = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def col():
    return build_collection()


@pytest.mark.live
def test_offtopic_query_has_low_relevancy(col):
    query = "how can I track my item"
    output, chunks = answer(query, col)

    tc = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=chunks,
    )

    metric = ContextualRelevancyMetric(threshold=0.5, model=EVAL_MODEL)
    metric.measure(tc)

    print(f"\nquery: {query}\nscore: {metric.score}\nreason: {metric.reason}")
    assert metric.score < 0.5


@pytest.mark.live
def test_in_scope_query_retrieves_relevant_chunks(col):
    query = "how long until I get my money back"
    output, chunks = answer(query, col)

    tc = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=chunks,
    )

    metric = ContextualRelevancyMetric(threshold=0.5, model=EVAL_MODEL)
    metric.measure(tc)

    print(f"\nscore: {metric.score}\nreason: {metric.reason}")
    assert metric.score >= 0.5


@pytest.mark.live
def test_recall_correct_fact_is_retrieved(col):
    query = "how long until I get my money back"
    output, chunks = answer(query, col)

    tc = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output="Refunds take 5 to 7 business days after the item is received.",
        retrieval_context=chunks,
    )

    metric = ContextualRecallMetric(threshold=0.7, model=EVAL_MODEL)
    metric.measure(tc)

    print(f"\nrecall: {metric.score}\nreason: {metric.reason}")
    assert metric.score >= 0.7


@pytest.mark.live
def test_precision_relevant_chunk_ranks_first(col):
    query = "how long until I get my money back"
    output, chunks = answer(query, col)

    tc = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output="Refunds take 5 to 7 business days after the item is received.",
        retrieval_context=chunks,
    )

    metric = ContextualPrecisionMetric(threshold=0.7, model=EVAL_MODEL)
    metric.measure(tc)

    print(f"\nprecision: {metric.score}\nreason: {metric.reason}")
    assert metric.score >= 0.7


@pytest.mark.live
def test_answer_is_faithful_to_chunks(col):
    query = "how long until I get my money back"
    output, chunks = answer(query, col)

    tc = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=chunks,
    )

    metric = FaithfulnessMetric(threshold=0.8, model=EVAL_MODEL)
    metric.measure(tc)

    print(f"\nfaithfulness: {metric.score}\nreason: {metric.reason}")
    assert metric.score >= 0.8


@pytest.mark.live
def test_answer_is_relevant_to_question(col):
    query = "how long until I get my money back"
    output, chunks = answer(query, col)

    tc = LLMTestCase(
        input=query,
        actual_output=output,
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL)
    metric.measure(tc)

    print(f"\nrelevancy: {metric.score}\nreason: {metric.reason}")
    assert metric.score >= 0.7


@pytest.mark.live
def test_out_of_scope_query_is_refused(col):
    query = "do you offer gift wrapping"
    output, chunks = answer(query, col)

    print(f"\nchunks: {chunks}\noutput: {output}")
    assert "don't have that information" in output.lower()
