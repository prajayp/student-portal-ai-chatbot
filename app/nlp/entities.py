"""Rule-based entity (slot) extraction for departments, semesters, course codes and programmes."""

import re
from dataclasses import dataclass
from typing import Callable

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}
_NUM = r"([1-8]|first|second|third|fourth|fifth|sixth|seventh|eighth|one|two|three|four|five|six|seven|eight)"

# Placeholder words substituted for entity mentions before classification, so the
# classifier learns "exam schedule for <department>" rather than specific values.
PLACEHOLDERS = {
    "course_code": "coursecode",
    "department": "department",
    "semester": "semester",
    "program": "programme",
}


@dataclass(frozen=True)
class _Rule:
    slot: str
    pattern: re.Pattern[str]
    normalize: Callable[[re.Match[str]], str | int]


def _const(value: str) -> Callable[[re.Match[str]], str]:
    return lambda _m: value


def _semester(m: re.Match[str]) -> int:
    raw = next(g for g in m.groups() if g).lower()
    return int(raw) if raw.isdigit() else ORDINALS[raw]


_I = re.IGNORECASE
# Order matters: earlier rules claim text first (e.g. "CS 301" is a course, not the CS department).
_RULES: list[_Rule] = [
    _Rule("course_code", re.compile(r"\b(CS|EC|EE|ME|CE|GE)\s?-?(\d{3})\b", _I),
          lambda m: f"{m.group(1).upper()}{m.group(2)}"),
    _Rule("program", re.compile(r"\bb\.?\s?tech\b|\bb\.?e\.?(?=\s|$)", _I), _const("BTECH")),
    _Rule("program", re.compile(r"\bm\.?\s?tech\b", _I), _const("MTECH")),
    _Rule("program", re.compile(r"\bm\.?b\.?a\b", _I), _const("MBA")),
    _Rule("program", re.compile(r"\bph\.?\s?d\b|\bdoctoral\b|\bdoctorate\b", _I), _const("PHD")),
    _Rule("department", re.compile(r"\bcomputer science(?: and engineering)?\b|\bcse\b|\bcomputers?\b", _I),
          _const("CSE")),
    _Rule("department", re.compile(r"\belectronics and communications?(?: engineering)?\b|\bece\b", _I),
          _const("ECE")),
    _Rule("department", re.compile(r"\belectrical and electronics(?: engineering)?\b|\beee\b|\belectrical\b", _I),
          _const("EEE")),
    _Rule("department", re.compile(r"\belectronics\b", _I), _const("ECE")),
    _Rule("department", re.compile(r"\bmechanical(?: engineering)?\b|\bmech\b", _I), _const("ME")),
    _Rule("department", re.compile(r"\bcivil(?: engineering)?\b", _I), _const("CE")),
    # Two-letter codes only when written in capitals ("ME" vs. the pronoun "me").
    _Rule("department", re.compile(r"\bCS\b"), _const("CSE")),
    _Rule("department", re.compile(r"\bEC\b"), _const("ECE")),
    _Rule("department", re.compile(r"\bEE\b"), _const("EEE")),
    _Rule("department", re.compile(r"\bME\b"), _const("ME")),
    _Rule("department", re.compile(r"\bCE\b"), _const("CE")),
    _Rule("semester", re.compile(rf"\b(?:semester|sem)\s*-?\s*{_NUM}\b", _I), _semester),
    _Rule("semester", re.compile(rf"\b{_NUM}(?:st|nd|rd|th)?\s*(?:semester|sem)\b", _I), _semester),
]


def _scan(text: str) -> tuple[dict[str, str | int], list[tuple[int, int, str]]]:
    """Return extracted slots and the (start, end, slot) spans they came from."""
    slots: dict[str, str | int] = {}
    spans: list[tuple[int, int, str]] = []
    for rule in _RULES:
        for m in rule.pattern.finditer(text):
            if any(m.start() < end and start < m.end() for start, end, _ in spans):
                continue
            spans.append((m.start(), m.end(), rule.slot))
            slots.setdefault(rule.slot, rule.normalize(m))
    return slots, sorted(spans)


def extract_entities(text: str) -> dict[str, str | int]:
    """Extract slot values from a user message, e.g. {"department": "CSE", "semester": 5}."""
    return _scan(text)[0]


def mask_entities(text: str) -> str:
    """Replace entity mentions with generic placeholder words."""
    _, spans = _scan(text)
    out, cursor = [], 0
    for start, end, slot in spans:
        out.append(text[cursor:start])
        out.append(f" {PLACEHOLDERS[slot]} ")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
