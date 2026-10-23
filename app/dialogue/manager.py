"""Dialogue manager: resolves each turn to an intent using the classifier plus session context."""

from dataclasses import dataclass
from enum import StrEnum

from app.dialogue.handlers import REGISTRY, Context, Response, chips
from app.dialogue.knowledge import intent_suggestions
from app.dialogue.roles import Role
from app.dialogue.state import DialogueState, SessionStore, Slots
from app.nlp.classifier import IntentClassifier, Prediction
from app.nlp.entities import extract_entities
from app.nlp.preprocess import is_bare_follow_up

SMALL_TALK = frozenset({"greeting", "goodbye", "thanks", "help", "fallback"})


class Resolution(StrEnum):
    CLASSIFIER = "classifier"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    FALLBACK = "fallback"
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class TurnResult:
    reply: str
    intent: str
    confidence: float
    suggestions: list[str]
    slots: Slots
    resolved_by: Resolution


class DialogueManager:
    """Stateful conversation handling on top of a stateless intent classifier."""

    def __init__(self, classifier: IntentClassifier, store: SessionStore, threshold: float) -> None:
        self.classifier = classifier
        self.store = store
        self.threshold = threshold

    def handle(self, session_id: str, role: Role, message: str) -> TurnResult:
        state = self.store.get(session_id, role)
        state.turns += 1
        entities = extract_entities(message)
        prediction = self.classifier.predict(message)
        intent, confidence, resolution = self._resolve(state, message, entities, prediction)

        if resolution is Resolution.FALLBACK:
            state.pending_slot = None
            return self._fallback(state, prediction, intent)

        spec = REGISTRY[intent]
        if role not in spec.roles:
            state.pending_slot = None
            allowed = " and ".join(sorted(r.value for r in spec.roles))
            reply = f"Sorry, {spec.title} is only available to {allowed} accounts. You're signed in as {role.value}."
            return TurnResult(reply, intent, confidence, chips(role), dict(state.slots), Resolution.RESTRICTED)

        state.remember(entities)
        response: Response = spec.handler(Context(role=role, slots=dict(state.slots), fresh=frozenset(entities)))
        state.pending_slot = response.awaiting_slot
        if intent not in SMALL_TALK:
            state.last_intent, state.last_confidence = intent, confidence
        return TurnResult(response.reply, intent, confidence, response.suggestions, dict(state.slots), resolution)

    def _resolve(
        self, state: DialogueState, message: str, entities: Slots, prediction: Prediction
    ) -> tuple[str, float, Resolution]:
        """Decide whether this turn answers a clarification, continues the last topic, or starts a new one."""
        uncertain = prediction.confidence < self.threshold
        follow_up_like = bool(entities) and (uncertain or is_bare_follow_up(message))

        if state.last_intent and follow_up_like:
            if state.pending_slot and state.pending_slot in entities:
                return state.last_intent, state.last_confidence, Resolution.CLARIFICATION
            if REGISTRY[state.last_intent].slots & entities.keys():
                return state.last_intent, state.last_confidence, Resolution.FOLLOW_UP

        if uncertain or prediction.intent == "fallback":
            return "fallback", prediction.confidence, Resolution.FALLBACK
        return prediction.intent, prediction.confidence, Resolution.CLASSIFIER

    def _fallback(self, state: DialogueState, prediction: Prediction, intent: str) -> TurnResult:
        """Out-of-scope or low-confidence reply with 'did you mean' suggestions."""
        role = state.role
        examples = intent_suggestions(role.value)
        guesses = [
            label for label, p in prediction.ranking
            if label not in SMALL_TALK and p >= 0.08 and role in REGISTRY[label].roles and label in examples
        ]
        if prediction.intent == "fallback" and prediction.confidence >= self.threshold:
            reply = "That's outside what I can help with. I answer questions about Example University: academics, admissions and campus services."
        else:
            reply = "Sorry, I didn't quite understand that."
            if guesses:
                reply += " Did you mean one of these?"
            else:
                reply += " Try rephrasing, or pick one of these:"
        suggestions = list(dict.fromkeys([examples[g] for g in guesses] + chips(role)))[:4]
        return TurnResult(reply, intent, prediction.confidence, suggestions, dict(state.slots), Resolution.FALLBACK)
