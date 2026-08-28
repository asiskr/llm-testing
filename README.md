# llm-testing

Prompt engineering experiments and an automated test suite for two LLM
applications: a **grounded FAQ bot** and a **RAG pipeline** over a returns
policy.

The point of the repo is not the bots — it is testing them. LLM output is
non-deterministic, so the interesting question is how you write assertions that
stay meaningful when the same input produces different text every run.

Model under test: `openai/gpt-oss-20b` via the [Groq](https://groq.com) API.

## Quick start

Requires **Python 3.12**.

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env      # then add your key
```

```bash
make test-offline    # 25 tests, ~3s, free, no API key needed
make test-live       # hits the API, costs quota
make lint            # ruff check + format check
make chat            # interactive FAQ bot
make rag             # interactive RAG query loop
```

`make help` lists every target.

## Configuration

`.env` needs **two** things:

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | The model under test. |
| `OPENAI_BASE_URL` + `OPENAI_API_KEY` | The **judge** model. DeepEval scores answers with an LLM that speaks the OpenAI protocol; pointing `OPENAI_BASE_URL` at Groq's OpenAI-compatible endpoint runs the judge on Groq too, so no OpenAI account is needed. Reuse the same Groq key. |

`.env` is gitignored. `.env.example` is the template.

## Project structure

```
src/llm_testing/
  llm_client.py      thin Groq wrapper; client built lazily so import needs no key
  returns_bot.py     FAQ bot: system prompt + ask_bot() — single source of truth
  rag.py             chunking, Chroma retrieval, grounded answer
  chat_cli.py        interactive FAQ bot          (llm-testing-chat)
  rag_cli.py         interactive RAG loop         (llm-testing-rag)

experiments/         one script per technique — exploratory, print-driven
data/policy.txt      the corpus the RAG pipeline retrieves over

tests/
  conftest.py        env loading + session-cached bot fixture
  offline/           no API calls, no credentials, runs in CI
  live/              real API calls, marked `live`, costs quota
```

Two rules hold this together:

- **The prompt lives in exactly one place.** `SYSTEM_PROMPT` is defined in
  `returns_bot.py` and imported by both the CLI and the tests, so the suite can
  never drift from what ships.
- **Import is side-effect free.** The Groq client is constructed on first use,
  not at import. That is what lets the offline suite collect and run with no
  `GROQ_API_KEY` at all — without it, CI cannot even import the package.

## The test suite

55 tests, split by what they cost.

### Offline — 25 tests, ~3s, no credentials

| Test file | Covers |
| --- | --- |
| `offline/test_returns_bot.py` | Prompt invariants (refusal string present, FAQ tags intact, `FAQ_CONTEXT` matches the prompt) and `ask_bot()` message assembly, with `chat()` mocked. |
| `offline/test_rag_retrieval.py` | Chunking is non-empty, 6 golden queries retrieve the right fact, and `MAX_DISTANCE` is pinned from **both** sides — in-scope queries must still retrieve, off-corpus queries must retrieve nothing. Local embeddings only — no LLM. |

> Mocking note: the mock patches `llm_testing.returns_bot.chat`, **not**
> `llm_testing.llm_client.chat`. `returns_bot` did `from ... import chat`, so it
> holds its own reference from import time; patching the defining module is a
> no-op that lets a real API call through while the test still passes. Patch
> where the function is *used*.

### Live — 30 tests, costs quota

| Test file | Covers |
| --- | --- |
| `live/test_faq_metrics.py` | AnswerRelevancy, Faithfulness, Hallucination, and a custom GEval brevity metric. |
| `live/test_faq_golden_set.py` | 11 fixed cases — 5 FAQ hits, 3 FAQ gaps, 3 off-topic. Substring match, no judge model, so this is the cheapest signal after a prompt change. |
| `live/test_prompt_injection.py` | 6 attacks: 3 fact injections, 3 role hijacks. |
| `live/test_rag_metrics.py` | ContextualRelevancy, ContextualRecall, ContextualPrecision, Faithfulness, AnswerRelevancy, and refusal on out-of-scope queries. |

### Assertion design

Two rules that stop tests failing on correct behaviour:

- **Assert the refusal, not the topic word.** `assert "poem" not in reply` goes
  red when the bot *correctly* says "I can't write a poem for you". Role-hijack
  tests assert `REFUSAL in reply` instead.
- **Give GEval steps, not a slogan.** A free-text `criteria` let the judge
  invent its own rubric and score a correct one-line answer 0.0 for "lacking
  reasoning and evidence". Explicit `evaluation_steps` that say *judge only
  length and formatting* fixed it.

## What each experiment covers

Run from the project root: `python -m experiments.<name>`

### Prompting

| Script | Technique |
| --- | --- |
| `basic_prompt.py` | The simplest prompt-and-response round trip. |
| `temperature_check.py` | The same prompt at `temperature=0` vs `1`, three runs each. |
| `json_output.py` | A vague instruction vs an explicit "valid JSON, no markdown" one, then parses and asserts. |
| `few_shot.py` | The same extraction with and without worked examples, sampled 5× at `temperature=1`. |
| `delimiters.py` | Fences input in `###` markers, marks it data-only, requires a fixed key set. |
| `flaky_assert.py` | An exact-equality assert on a sampled response — fails intermittently, which is the point. |
| `multi_turn.py` | Asks "What is my name?" with no prior turn: the model has no memory the caller didn't supply. |

### Retrieval

| Script | Technique |
| --- | --- |
| `keyword_fail.py` | Keyword matching finds nothing for "how long until I get my money back" — the motivation for embeddings. |
| `embed_demo.py` | Cosine similarity between a query and one document. |
| `compare_docs.py` | The same query scored against several documents. |
| `chunk_demo.py` | Naive fixed-size character chunking with overlap, and why it splits sentences badly. |
| `check_chunks.py` | The sentence-level chunks the pipeline actually uses. |
| `chroma_demo.py` | A persistent Chroma collection: add, count, query. |
| `see_distances.py` | Distances for in-scope vs off-topic queries — how `MAX_DISTANCE` was chosen. |

## Patterns

| # | Pattern | Why it works | Test it enables |
| --- | --- | --- | --- |
| 1 | Ask for JSON, not prose | Free text reformats between runs; JSON keeps a fixed shape | `json.loads()` parses; required keys present |
| 2 | Show examples, don't describe | The model copies a demonstrated format more reliably than a described one | Shape identical across N runs |
| 3 | Fence the data in `###` markers | Marking input as data-only stops the model obeying instructions hidden inside it | Injected "reply BANANA" is ignored |
| 4 | State the output contract in the prompt | Without it the model drops keys entirely; with it the key set is fixed | Both keys present, `null` when absent |
| 5 | Compare prompts at `temperature=1` | Temperature 0 hides weak prompts — differences only surface under sampling | Pass-rate over N runs, not a single pass/fail |
| 6 | Give the model an explicit escape hatch | "Say exactly *I don't have that information*" turns a refusal into a matchable string | Gap and off-topic cases assert one substring |
| 7 | Keep the prompt in one place | A prompt duplicated into the tests means you test a copy, not the thing that ships | The suite imports `SYSTEM_PROMPT` from the package |
| 8 | Declare the user turn to be data | The user turn can't be fenced the way `###` input can, so the system prompt must say it outright | Injection tests assert the injected fact never appears |
| 9 | Drop low-scoring chunks before generating | If retrieval returns nothing within the distance threshold, refuse instead of feeding the model noise | Off-corpus queries return no context at all |
| 10 | Pin a tuned constant from both sides | A threshold with only a lower-bound test drifts upward until it stops filtering; measured bands make the safe range explicit | `MAX_DISTANCE` has one test that fails if it is tightened and one that fails if it is loosened |

## Evaluation policy

| Metric | Threshold | N | Pass-rate |
|---|---|---|---|
| Faithfulness | 0.8 | 5 | 5/5 |
| ContextualRecall | 0.7 | 5 | 4/5 |
| ContextualPrecision | 0.7 | 5 | 4/5 |
| AnswerRelevancy | 0.7 | 5 | 4/5 |
| ContextualRelevancy | 0.5 | 3 | 3/3 |

**Why these numbers**

- Faithfulness requires 5/5 — a single hallucinated run means a user could see a
  false answer. Other metrics tolerate one flaky run.
- Thresholds are set mid-range, not at the observed score. Judge scores vary
  between runs, so a threshold at the boundary produces false failures.
- Retry is limited to rate-limit errors only. Retrying on any failure would mask
  genuine quality regressions.

**Test tiers**

| Tier | What | When |
|---|---|---|
| Offline | chunking, retrieval, prompt invariants, message assembly | every commit (CI) |
| Live smoke | one query per metric, N=1 | every PR, manually |
| Full eval | all goldens × all metrics, N=5 | nightly |

Live evals are rate-limited by the free tier: Groq allows 8000 tokens/min, and a
full live run burns that in under a minute. `pyproject.toml` retries **only** on
rate-limit errors (`--only-rerun RateLimitError`), never on assertion failures,
so a real regression still shows up red.

## Roadmap

- [x] Offline tests that run in CI with no credentials
- [x] Prompt-injection tests
- [x] Packaged layout, `pyproject.toml`, ruff, pre-commit, GitHub Actions
- [ ] Pass-rate assertions over N runs (`pytest-repeat` is installed)
- [ ] Pydantic schema validation in place of hand-written key checks
- [ ] Nightly scheduled workflow for the full eval matrix
- [ ] Hybrid retrieval (keyword + vector) and a reranking step

## License

MIT — see [LICENSE](LICENSE).
