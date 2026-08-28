"""Retrieval-augmented generation over the ShopEasy returns policy.

Sentence-level chunks are embedded into an in-memory Chroma collection;
retrieval drops anything past MAX_DISTANCE so an off-topic question returns
no context at all and the caller can refuse instead of guessing.
"""

from pathlib import Path

import chromadb

from llm_testing.llm_client import ask

# Resolved from the package rather than the working directory, so the module
# behaves the same no matter where it is imported from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "data" / "policy.txt"

# Chroma returns squared L2 distance; past this the chunk is noise.
MAX_DISTANCE = 1.6

REFUSAL = "I don't have that information."


def load_chunks(path=DEFAULT_POLICY_PATH):
    with open(path) as f:
        text = f.read()

    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    return sentences


def build_collection():
    chunks = load_chunks()
    ids = [f"policy-{i}" for i in range(len(chunks))]
    client = chromadb.Client()
    col = client.get_or_create_collection("policy")
    col.add(documents=chunks, ids=ids)
    return col


def retrieve(col, query, k=2):
    res = col.query(query_texts=[query], n_results=k)

    ids = res["ids"][0]
    docs = res["documents"][0]
    dists = res["distances"][0]

    keep_ids, keep_docs = [], []

    for _id, doc, dist in zip(ids, docs, dists, strict=True):
        if dist <= MAX_DISTANCE:
            keep_ids.append(_id)
            keep_docs.append(doc)

    return keep_ids, keep_docs


def answer(query, col):
    _ids, chunks = retrieve(col, query)
    if not chunks:
        return REFUSAL, []
    context = "\n\n".join(chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: I don't have that information.
Keep the answer to one sentence.

<context>
{context}
</context>

Question: {query}"""
    return ask(prompt), chunks
