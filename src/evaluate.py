from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.load_data import load_sample_resumes
from src.models.classifier import build_classifier


def evaluate_sample_classifier() -> dict:
    """Evaluate on synthetic examples. Metrics are illustrative and not real-world evidence."""
    df = load_sample_resumes()
    x, y = df["resume_text"].astype(str), df["category"].astype(str).str.strip()
    # Small synthetic data: use a deterministic stratified split where class counts allow it.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=42, stratify=y
    )
    model = build_classifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    labels = sorted(y.unique())
    return {"model": model, "accuracy": float(accuracy_score(y_test, pred)),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
            "labels": labels, "test_size": len(y_test), "train_size": len(y_train)}
