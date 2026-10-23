"""Shared fixtures: one trained classifier per test session and a fresh dialogue manager per test."""

import pytest

from app import config
from app.dialogue.manager import DialogueManager
from app.dialogue.roles import Role
from app.dialogue.state import SessionStore
from app.nlp.classifier import IntentClassifier
from app.nlp.train import load_dataset


@pytest.fixture(scope="session")
def classifier() -> IntentClassifier:
    return IntentClassifier.train(*load_dataset())


@pytest.fixture
def manager(classifier: IntentClassifier) -> DialogueManager:
    return DialogueManager(classifier, SessionStore(), config.CONFIDENCE_THRESHOLD)


@pytest.fixture
def chat(manager: DialogueManager):
    """Send messages in one session: chat("hello", role=Role.FACULTY)."""

    def send(message: str, role: Role = Role.STUDENT, session: str = "test-session"):
        return manager.handle(session, role, message)

    return send
