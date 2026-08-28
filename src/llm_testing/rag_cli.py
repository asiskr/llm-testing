"""Interactive RAG query loop over the returns policy.

    llm-testing-rag          (after `pip install -e .`)
    python -m llm_testing.rag_cli

Prints the retrieved chunks alongside the answer, so you can see whether a bad
answer came from bad retrieval or bad generation. Type 'exit' to quit.
"""

from llm_testing.rag import answer, build_collection, retrieve


def main():
    col = build_collection()
    print("Ask a question about the returns policy. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        ids, docs = retrieve(col, query)

        print("\n--- RETRIEVED CHUNKS ---")
        if not docs:
            print("(none within MAX_DISTANCE)\n")
        for chunk_id, doc in zip(ids, docs, strict=True):
            print(f"[{chunk_id}] {doc}\n")

        reply, _chunks = answer(query, col)
        print("--- ANSWER ---")
        print(reply, "\n")


if __name__ == "__main__":
    main()
