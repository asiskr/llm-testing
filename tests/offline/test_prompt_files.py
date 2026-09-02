"""The promptfoo harness reads prompts from prompts/*.json, not from the
application. If SYSTEM_PROMPT is edited and the file isn't regenerated, the
harness silently evaluates a prompt that is no longer in production.
Offline and free - no model call.
"""

import json
from pathlib import Path

from llm_testing.returns_bot import SYSTEM_PROMPT

V1 = Path("prompts/returns_bot_v1.json")


def test_v1_prompt_file_matches_system_prompt():
    saved = json.loads(V1.read_text())[0]["content"]

    assert saved == SYSTEM_PROMPT, (
        "prompts/returns_bot_v1.json is out of sync with SYSTEM_PROMPT. Regenerate it - see README."
    )
