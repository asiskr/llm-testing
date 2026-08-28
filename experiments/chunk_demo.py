with open("data/policy.txt") as f:
    text = f.read()

print("poore text me kitne letters:", len(text))

CHUNK_SIZE = 200
OVERLAP = 50

step = CHUNK_SIZE - OVERLAP

chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), step)]

print("kitne chunks bane:", len(chunks), "\n")

for i, c in enumerate(chunks):
    print(f"--- chunk {i} ---")
    print(c)
    print()
