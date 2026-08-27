import chromadb
from src.llm_client import ask


def load_chunks(path="data/policy.txt"):
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


MAX_DISTANCE = 1.6

def retrieve(col, query, k=2):
    res = col.query(query_texts=[query], n_results=k)

    ids   = res["ids"][0]
    docs  = res["documents"][0]
    dists = res["distances"][0]

    keep_ids, keep_docs = [], []

    for _id, doc, dist in zip(ids, docs, dists):
        if dist <= MAX_DISTANCE:
            keep_ids.append(_id)
            keep_docs.append(doc)

    return keep_ids, keep_docs


def answer(query, col):
    _ids, chunks = retrieve(col, query)
    if not chunks:
        return "I don't have that information.", []
    context = "\n\n".join(chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: I don't have that information.
Keep the answer to one sentence.

<context>
{context}
</context>

Question: {query}"""
    return ask(prompt), chunks


if __name__ == "__main__":
    col = build_collection()
    q = "how long until I get my money back"
    ids, chunks = retrieve(col, q)
    for i, c in zip(ids, chunks):
        print(f"[{i}] {c}\n")
    print(answer(q, col))