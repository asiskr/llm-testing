import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

from experiments.rag_app import build_collection, answer

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