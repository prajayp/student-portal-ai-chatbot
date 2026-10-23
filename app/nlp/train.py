"""Train the intent classifier, report held-out metrics and save the model.

Usage: python -m app.nlp.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app import config
from app.nlp.classifier import IntentClassifier, build_pipeline

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: Path = config.INTENTS_PATH) -> tuple[list[str], list[str]]:
    """Flatten intents.json into parallel lists of phrases and intent tags."""
    intents = json.loads(path.read_text(encoding="utf-8"))["intents"]
    texts = [p for intent in intents for p in intent["patterns"]]
    labels = [intent["tag"] for intent in intents for _ in intent["patterns"]]
    return texts, labels


def split_dataset(texts: list[str], labels: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Deterministic stratified train/test split used by training and tests alike."""
    return train_test_split(texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)


def evaluate() -> tuple[dict, str]:
    """Fit on the training split and score on the held-out split plus 5-fold CV."""
    texts, labels = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(texts, labels)
    model = IntentClassifier.train(x_train, y_train)
    probs = model.pipeline.predict_proba(x_test)
    y_pred = model.pipeline.classes_[probs.argmax(axis=1)]
    top = probs.max(axis=1)
    correct = y_pred == np.asarray(y_test)
    kept = top >= config.CONFIDENCE_THRESHOLD

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(build_pipeline(), texts, labels, cv=cv, scoring="accuracy")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_intents": len(set(labels)),
        "n_examples": len(texts),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "cv5_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv5_accuracy_std": round(float(cv_scores.std()), 4),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        # Share of correct test predictions answered directly vs. errors caught by the fallback.
        "correct_above_threshold": round(float(kept[correct].mean()), 4),
        "errors_below_threshold": round(float((~kept[~correct]).mean()), 4) if (~correct).any() else None,
    }
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)
    return metrics, report


def main() -> None:
    metrics, report = evaluate()
    print(json.dumps(metrics, indent=2))
    print(report)

    texts, labels = load_dataset()
    IntentClassifier.train(texts, labels).save(config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    config.REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved model (refit on all {len(texts)} examples) to {config.MODEL_PATH.relative_to(config.PROJECT_DIR)}")


if __name__ == "__main__":
    main()
