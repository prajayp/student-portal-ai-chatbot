"""TF-IDF + logistic regression intent classifier."""

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.nlp.preprocess import normalize_batch


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    ranking: list[tuple[str, float]]


def build_pipeline() -> Pipeline:
    """Word (1-2) and character (2-5) n-gram TF-IDF features into multinomial logistic regression."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
    ])
    return Pipeline([
        ("normalize", FunctionTransformer(normalize_batch)),
        ("features", features),
        ("clf", LogisticRegression(C=20.0, max_iter=5000, class_weight="balanced")),
    ])


class IntentClassifier:
    """Thin wrapper around a fitted pipeline that returns ranked predictions."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> "IntentClassifier":
        pipeline = build_pipeline()
        pipeline.fit(texts, labels)
        return cls(pipeline)

    @classmethod
    def load(cls, path: Path) -> "IntentClassifier":
        return cls(joblib.load(path))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @property
    def labels(self) -> list[str]:
        return list(self.pipeline.classes_)

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        probs = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.pipeline.classes_, probs), key=lambda p: p[1], reverse=True)
        ranking = [(str(label), float(p)) for label, p in ranked[:top_k]]
        return Prediction(intent=ranking[0][0], confidence=ranking[0][1], ranking=ranking)
