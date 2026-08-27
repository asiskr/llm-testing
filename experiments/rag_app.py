import chromadb
from src.llm_client import ask

CHUNK_SIZE, OVERLAP = 200, 50


def load_chunks(path="data/policy.txt"):
    with open(path) as f:
        text = f.read()
    step = CHUNK_SIZE - OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]


def build_collection():
    chunks = load_chunks()
    ids = [f"policy-{i}" for i in range(len(chunks))]
    client = chromadb.Client()
    col = client.get_or_create_collection("policy")
    col.add(documents=chunks, ids=ids)
    return col


def retrieve(col, query, k=2):
    res = col.query(query_texts=[query], n_results=k)
    return res["ids"][0], res["documents"][0]


def answer(query, col):
    _ids, chunks = retrieve(col, query)
    context = "\n\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: I don't have that information.
Keep the answer to one sentence.

<context>
{context}
</context>

Question: {query}"""
    return ask(prompt)


if __name__ == "__main__":
    col = build_collection()
    q = "how long until I get my money back"
    ids, chunks = retrieve(col, q)
    for i, c in zip(ids, chunks):
        print(f"[{i}] {c}\n")
    print(answer(q, col))