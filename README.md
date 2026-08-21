# llm-testing

Hands-on experiments in **prompt engineering** and testing LLM behaviour.

Each script in `experiments/` isolates one prompting technique and shows how the
model's output changes when the prompt changes. The goal is to build intuition
for *why* a prompt succeeds or fails, and to practise asserting on
non-deterministic output.

Model: `openai/gpt-oss-20b`, served via the [Groq](https://groq.com) API.

## Setup

Requires **Python 3.12**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

`.env` is gitignored — never commit your key.

## Running the experiments

Run from the **project root** (the scripts add the root to `sys.path`):

```bash
python experiments/basic_prompt.py
```

Every script makes live API calls, so each run costs quota and output will vary
between runs.

## What each experiment covers

| Script | Technique | What it shows |
| --- | --- | --- |
| `basic_prompt.py` | Zero-shot prompting | The simplest possible prompt-and-response round trip. |
| `temperature_check.py` | Sampling / determinism | The same prompt at `temperature=0` vs `temperature=1`, three runs each. |
| `json_output.py` | Structured output | A vague instruction vs an explicit "return valid JSON, no markdown" instruction — then parses and asserts on the result. |
| `few_shot.py` | Zero-shot vs few-shot | The same extraction task with and without worked examples, sampled five times at `temperature=1` to expose consistency differences. |
| `delimiters.py` | Delimiters & data separation | Fences input between `###` markers, marks it as data-only, and requires a fixed key set with `null` for missing values. |

## Project structure

```
src/llm_client.py     # thin Groq wrapper: ask(prompt, temperature=0)
experiments/          # one script per prompting technique
tests/                # (empty) automated test suite — see Roadmap
```

## Roadmap

The experiments are exploratory scripts, not an automated test suite — they
print results and assert inline. Planned next:

- Port the experiments to `pytest` under `tests/`, split into mocked (offline,
  CI-safe) and live (`@pytest.mark.live`) runs
- Pass-rate assertions over N runs instead of single asserts, to handle
  non-determinism
- Pydantic schema validation in place of hand-written key checks
- System / role prompts and multi-turn conversation support in `ask()`
- Additional techniques: chain-of-thought, grounding and refusal, prompt
  injection, self-consistency, prompt chaining
