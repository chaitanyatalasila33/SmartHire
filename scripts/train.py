import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import MODELS_DIR, ROOT_DIR
from src.data.load_data import load_sample_resumes
from src.models.classifier import build_classifier


def main():
    df = load_sample_resumes()
    x = df["resume_text"].astype(str)
    y = df["category"].astype(str).str.strip()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
    model = build_classifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "resume_classifier.joblib")
    report = {
        "data_note": "Metrics are from the synthetic demonstration dataset, not a real-world benchmark.",
        "random_state": 42, "train_size": len(x_train), "test_size": len(x_test),
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "labels": sorted(y.unique()),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist(),
    }
    (ROOT_DIR / "reports" / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))
    print(f"Saved model to {MODELS_DIR / 'resume_classifier.joblib'}")

if __name__ == "__main__":
    main()
