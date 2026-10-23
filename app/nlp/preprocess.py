"""Text normalisation shared by training and inference."""

import re
from collections.abc import Iterable

from app.nlp.entities import PLACEHOLDERS, mask_entities

_NON_WORD = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")

# Words that carry no intent on their own; a message made only of these plus
# entities ("and for ECE?", "what about semester 5") is treated as a follow-up.
FOLLOW_UP_WORDS = frozenset(
    "and for what what's whats about how the same in of with then also ok okay please "
    "is it its now sem semester a an".split()
)


def normalize(text: str) -> str:
    """Mask entities, lowercase and strip punctuation."""
    text = mask_entities(text).lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def normalize_batch(texts: Iterable[str]) -> list[str]:
    """Vectorised form of :func:`normalize` for use inside a scikit-learn pipeline."""
    return [normalize(t) for t in texts]


def is_bare_follow_up(text: str) -> bool:
    """True when the message has no content words beyond entities and follow-up fillers."""
    placeholders = set(PLACEHOLDERS.values())
    tokens = normalize(text).split()
    return all(t in FOLLOW_UP_WORDS or t in placeholders for t in tokens)
