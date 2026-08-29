"""Offline tests for the retrieval half of the RAG pipeline.

Chroma embeds locally, so nothing here calls an LLM and nothing needs a key.

These tests pin MAX_DISTANCE from both sides. Measured top-1 distances against
data/policy.txt:

    in-scope        0.41 - 1.13
    out-of-scope    1.11 - 1.44   (plausible returns questions the policy
                                   does not answer)
    off-corpus      1.84 - 1.98   (nothing in the policy is even related)

The in-scope and out-of-scope bands overlap, so no threshold can separate
them - which is why there is no offline test that out-of-scope queries
retrieve nothing. They do retrieve chunks, and refusing them is the
generation step's job, covered by
tests/live/test_rag_metrics.py::test_out_of_scope_query_is_refused.

What MAX_DISTANCE can do is keep unrelated text away from the model, and that
is what the two boundary tests below hold in place.
"""

import pytest

from llm_testing.rag import MAX_DISTANCE, build_collection, load_chunks, retrieve

# In-scope: the policy answers these, so the named fact must come back.
GOLDEN = [
    ("How long can i expect my refund", "5 to 7 business days"),
    ("what can be the last day of return", "30 days of delivery."),
    ("on what source of payement will I get my refund", "original payment method only."),
    ("Who pays the return shipping?", "The customer pays for return shipping"),
    ("Can I return the sale item?", "final sale"),  
    ("I missed my return date, can I reschedule it", "30 days"),
]

# Off-corpus: no sentence in the policy is topically close, so every candidate
# falls beyond MAX_DISTANCE and retrieve() hands back nothing at all.
OFF_CORPUS = [
    "what is the capital of france",
    "write me a poem about dogs",
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


@pytest.mark.parametrize("query", [q for q, _ in GOLDEN])
def test_max_distance_admits_every_in_scope_query(col, query):
    """Lower bound on the threshold.

    The weakest in-scope query sits at ~1.13. Tighten MAX_DISTANCE past that
    to suppress out-of-scope questions and real ones start retrieving nothing.
    """
    _ids, docs = retrieve(col, query)
    assert docs, f"in-scope query retrieved nothing at MAX_DISTANCE={MAX_DISTANCE}"


@pytest.mark.parametrize("query", OFF_CORPUS)
def test_off_corpus_query_retrieves_nothing(col, query):
    """Upper bound on the threshold.

    Off-corpus queries sit at ~1.84+. Loosen MAX_DISTANCE past that and
    unrelated policy text starts reaching the model as context.
    """
    _ids, docs = retrieve(col, query)
    assert docs == [], f"unrelated query retrieved context: {docs}"

@pytest.mark.xfail(
    reason="k=2 is too small: 'return date' pulls policy-1 and policy-3, so the "
    "sale-items rule never makes the cut. Fix is k=3 or hybrid search.",
    strict=True,
)
def test_known_gap_sale_item_phrasing(col):
    """Same fact as 'Can I return the sale item?' which passes — different
    phrasing, wrong chunks. Golden sets only catch what you thought to ask."""
    _ids, docs = retrieve(col, "what is the return date for my sale item")
    assert "final sale" in " ".join(docs)
