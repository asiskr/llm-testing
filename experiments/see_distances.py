from experiments.rag_app import build_collection

col = build_collection()

QUERIES = [
    "how long until I get my money back", 
    "who pays return shipping",        
    "how can I track my item",           
    "can I change my address for the return",
    "what is the capital of france",  
    "I missed my return date, can I reschedule it",
]

for q in QUERIES:
    res = col.query(query_texts=[q], n_results=2)
    dists = [round(d, 3) for d in res["distances"][0]]
    print(dists, "|", q)