from llm_testing.rag import load_chunks

chunks = load_chunks()

print("kitne chunks:", len(chunks), "\n")

for i, c in enumerate(chunks):
    print(f"[{i}] {c}")
