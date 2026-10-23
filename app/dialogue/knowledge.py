"""Read-only access to the fictional Example University dataset."""

import json
from functools import lru_cache
from typing import Any

from app import config


@lru_cache(maxsize=1)
def university() -> dict[str, Any]:
    """Load and cache ``university.json``."""
    return json.loads(config.UNIVERSITY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def intent_suggestions(role: str) -> dict[str, str]:
    """Map each intent to the example question shown as a suggestion chip for ``role``.

    A suggestion in intents.json is either one string or a {role: text} mapping.
    """
    intents = json.loads(config.INTENTS_PATH.read_text(encoding="utf-8"))["intents"]
    out: dict[str, str] = {}
    for intent in intents:
        text = intent["suggestion"]
        if isinstance(text, dict):
            text = text.get(role, "")
        if text:
            out[intent["tag"]] = text
    return out
