import pytest
from experiments.rag_app import load_chunks, build_collection, retrieve


@pytest.fixture(scope="module")
def col():
    return build_collection()


def test_chunks_are_not_empty():
    chunks = load_chunks()
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


def test_refund_query_retrieves_the_refund_chunk(col):
    _ids, docs = retrieve(col, "how long until I get my money back")
    joined = " ".join(docs)
    assert "5 to 7 business days" in joined


def test_offtopic_query_does_not_retrieve_refund_facts(col):
    _ids, docs = retrieve(col, "do you offer gift wrapping")
    joined = " ".join(docs)
    assert "5 to 7 business days" not in joined