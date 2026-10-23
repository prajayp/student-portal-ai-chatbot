"""Intent classifier and entity extraction."""

import pytest
from sklearn.metrics import accuracy_score

from app.nlp.classifier import IntentClassifier
from app.nlp.entities import extract_entities, mask_entities
from app.nlp.preprocess import is_bare_follow_up
from app.nlp.train import load_dataset, split_dataset

MIN_TEST_ACCURACY = 0.90


def test_held_out_accuracy_above_threshold() -> None:
    texts, labels = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(texts, labels)
    model = IntentClassifier.train(x_train, y_train)
    accuracy = accuracy_score(y_test, model.pipeline.predict(x_test))
    assert accuracy >= MIN_TEST_ACCURACY, f"held-out accuracy {accuracy:.3f}"


def test_every_intent_has_enough_examples() -> None:
    _, labels = load_dataset()
    counts = {label: labels.count(label) for label in set(labels)}
    assert len(counts) == 25
    assert min(counts.values()) >= 30, counts


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("When are the end semester exams for ECE?", "exam_schedule"),
        ("show me the class timetable for mechanical semester 3", "timetable"),
        ("how do I apply for revaluation", "results"),
        ("what documents should I bring for admission", "admission_documents"),
        ("is there a hostel for first year students", "hostel"),
        ("my laptop won't connect to the wifi", "wifi"),
        ("I need to see a doctor", "medical"),
        ("upload internal marks for CS501", "grade_submission"),
        ("how many applications have we received this year", "admission_stats"),
        ("tell me a joke", "fallback"),
    ],
)
def test_unseen_phrasings(classifier: IntentClassifier, message: str, intent: str) -> None:
    assert classifier.predict(message).intent == intent


def test_prediction_ranking_is_sorted_probabilities(classifier: IntentClassifier) -> None:
    prediction = classifier.predict("library timings")
    probs = [p for _, p in prediction.ranking]
    assert probs == sorted(probs, reverse=True)
    assert 0 < prediction.confidence <= 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("exam schedule for CSE", {"department": "CSE"}),
        ("timetable for electronics 5th sem", {"department": "ECE", "semester": 5}),
        ("what about semester five?", {"semester": 5}),
        ("tell me about cs 501", {"course_code": "CS501"}),
        ("fees for B.Tech mechanical", {"program": "BTECH", "department": "ME"}),
        ("electrical and electronics engineering", {"department": "EEE"}),
        ("can you tell me about the library", {}),  # lowercase "me" is not Mechanical
    ],
)
def test_entity_extraction(text: str, expected: dict) -> None:
    assert extract_entities(text) == expected


def test_masking_replaces_entities_with_placeholders() -> None:
    assert mask_entities("exam for CSE sem 5").split() == ["exam", "for", "department", "semester"]


@pytest.mark.parametrize(
    ("text", "bare"),
    [("and for ECE?", True), ("what about semester 5", True), ("ECE", True), ("hostel fees for ECE", False)],
)
def test_bare_follow_up_detection(text: str, bare: bool) -> None:
    assert is_bare_follow_up(text) is bare
