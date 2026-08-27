import chromadb

# client = chromadb.Client()
client = chromadb.PersistentClient(path="chroma_db")

# col = client.create_collection("faq")
col = client.get_or_create_collection("faq")
print("Counts of number of docs:", col.count())


col.add(
    documents=[
        "Refunds are processed within 5 to 7 business days.",
        "Our store is open from 9am to 9pm every day.",
        "Damaged items can be exchanged at any outlet.",
    ],
    ids=["d1", "d2", "d3"],
)
print("after add:", col.count())

res = col.query(query_texts=["how long until I get my money back"], n_results=3)

for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(round(dist, 3), "|", doc)