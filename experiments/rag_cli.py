from experiments.rag_app import build_collection, retrieve, answer

col = build_collection()

print("Sawaal poocho. Nikalne ke liye 'exit'\n")

while True:
    query = input("You: ").strip()

    if not query:
        continue
    if query.lower() == "exit":
        break

    ids, docs = retrieve(col, query)

    print("\n--- CHUNKS ---")
    for i, d in zip(ids, docs):
        print(f"[{i}] {d}\n")

    print("--- ANSWER ---")
    ans, _chunks = answer(query, col)
    print(ans, "\n")