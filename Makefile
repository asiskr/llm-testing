.PHONY: help install test test-offline test-live lint format chat rag clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dev tooling into the active venv
	pip install -e ".[dev]"

test-offline:  ## Fast, free, no API key needed
	pytest -m "not live"

test-live:  ## Hits the Groq API — costs quota
	pytest -m live

test:  ## Everything
	pytest

lint:  ## Check style and imports
	ruff check .
	ruff format --check .

format:  ## Autofix and format
	ruff check --fix .
	ruff format .

chat:  ## Interactive FAQ bot
	python -m llm_testing.chat_cli

rag:  ## Interactive RAG query loop
	python -m llm_testing.rag_cli

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -path ./.venv-new -prune -o -name __pycache__ -type d -exec rm -rf {} +
