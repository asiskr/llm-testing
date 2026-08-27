from chromadb.utils import embedding_functions
import numpy as np

ef = embedding_functions.DefaultEmbeddingFunction()

def similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

query = "how long until I get my money back"
doc   = "Refunds are processed within 5 to 7 business days."

v1 = ef([query])[0]
v2 = ef([doc])[0]

print("query :", query)
print("doc   :", doc)
print("score :", round(similarity(v1, v2), 3))