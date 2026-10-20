from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.features.text_features import make_tfidf_vectorizer


@dataclass
class ClassifierResult:
    model: Pipeline
    accuracy: float
    report: dict
    confusion_matrix: list
    labels: list


def build_classifier() -> Pipeline:
    return Pipeline([
        ("tfidf", make_tfidf_vectorizer()),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])


def predict_category(model: Pipeline, resume_text: str) -> tuple[str, float]:
    probabilities = model.predict_proba([resume_text])[0]
    idx = int(probabilities.argmax())
    return str(model.classes_[idx]), float(probabilities[idx])
