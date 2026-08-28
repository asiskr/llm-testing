DOCS = [
    "Refunds are processed within 5 to 7 business days.",
    "Our store is open from 9am to 9pm every day.",
    "Damaged items can be exchanged at any outlet.",
]

QUERY = "how long until I get my money back"

hits = [d for d in DOCS if any(w in d.lower().split() for w in QUERY.lower().split())]

print("QUERY :", QUERY)
print("HITS  :", hits if hits else "Nothing to find")
