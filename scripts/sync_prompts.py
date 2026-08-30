"""Regenerate prompts/returns_bot_v1.json from SYSTEM_PROMPT.

The promptfoo harness reads prompt files, not the application, so editing
SYSTEM_PROMPT leaves the harness on a stale prompt. test_prompt_files.py
catches that; this script is the fix it tells you to run.
"""

import json
from pathlib import Path

from llm_testing.returns_bot import SYSTEM_PROMPT

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "{{question}}"},
]

Path("prompts/returns_bot_v1.json").write_text(json.dumps(messages, indent=2))