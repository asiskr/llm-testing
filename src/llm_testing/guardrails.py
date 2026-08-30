"""Output guardrails: checks that run on the model's reply, not instructions
given to the model. The system prompt asks; these enforce.

Rule 4 of SYSTEM_PROMPT caps replies at 2 sentences. Nothing verified that
until now - the model happened to comply.

Known limit: the sentence regex over-counts abbreviations ("Mr. Sharma",
"e.g.") because it splits on any .!? followed by whitespace. Acceptable here
because the FAQ contains none; a real fix needs an NLP sentence splitter,
which is a dependency this repo does not want.
"""

import re

MAX_SENTENCES = 2

# Split on . ! ? followed by whitespace or end of string. Keeps "5-7" and
# "v2.0" intact because those have no space after the dot.
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def count_sentences(text):
    return len(_SENTENCE_END.findall(text.strip()))


def within_sentence_limit(text, limit=MAX_SENTENCES):
    return count_sentences(text) <= limit