import pytest
from experiments.rag_app import load_chunks, build_collection, retrieve
import re

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

GOLDEN = [
    ("How long can i expect my refund", "5 to 7 business days"),
    ("what can be the last day of return", "30 days of delivery."),
    ("on what source of payement will I get my refund", "original payment method only."),
    ("Who pays the return shipping?", "The customer pays for return shipping"),
    ("Can I return the sale item?", "final sale"),
    ("I missed my return date, can I reschedule it", "30 days"),
]
OUT_OF_SCOPE = [
    "will I get my app coins back",
    "can I change my address for the return",
    "can I get the delivery agent's number",
    "how can I track my item",
]

@pytest.fixture(scope="module")
def col():
    return build_collection()


def test_chunks_are_not_empty():
    chunks = load_chunks()
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


@pytest.mark.parametrize("query, expected_fact", GOLDEN)
def test_retrieval_finds_the_right_fact(col, query, expected_fact):
    _ids, docs = retrieve(col, query)
    assert expected_fact in " ".join(docs)