"""Per-session dialogue state and an in-memory, TTL-bounded session store."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app import config
from app.dialogue.roles import Role

Slots = dict[str, str | int]


@dataclass
class DialogueState:
    """Everything the bot remembers about one conversation."""

    role: Role
    last_intent: str | None = None
    last_confidence: float = 0.0
    slots: Slots = field(default_factory=dict)
    pending_intent: str | None = None
    pending_slot: str | None = None
    turns: int = 0
    updated_at: float = field(default_factory=time.monotonic)

    def remember(self, new_slots: Slots) -> None:
        """Merge freshly extracted slots into memory.

        A new department or semester invalidates a remembered course code (it most
        likely belonged to the previous context), and a course code implies its department.
        """
        if ("department" in new_slots or "semester" in new_slots) and "course_code" not in new_slots:
            self.slots.pop("course_code", None)
        self.slots.update(new_slots)

    def clear_pending(self) -> None:
        self.pending_intent = None
        self.pending_slot = None


class SessionStore:
    """Thread-safe in-memory store keyed by session id with idle expiry and LRU eviction."""

    def __init__(self, ttl_seconds: int = config.SESSION_TTL_SECONDS, max_sessions: int = config.MAX_SESSIONS) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions: OrderedDict[str, DialogueState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str, role: Role) -> DialogueState:
        """Return the session's state, starting fresh if it is new, expired or the role changed."""
        now = time.monotonic()
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or now - state.updated_at > self._ttl or state.role != role:
                state = DialogueState(role=role)
            state.updated_at = now
            self._sessions[session_id] = state
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
            return state

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)
