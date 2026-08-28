import numpy as np
from chromadb.utils import embedding_functions

ef = embedding_functions.DefaultEmbeddingFunction()


def similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


DOCS = [
    "Refunds are processed within 5 to 7 business days.",
    "Our store is open from 9am to 9pm every day.",
    "Damaged items can be exchanged at any outlet.",
]

query = "how long until I get my money back"

qv = ef([query])[0]

print("QUERY:", query, "\n")

for d in DOCS:
    dv = ef([d])[0]
    print(round(similarity(qv, dv), 3), "|", d)
