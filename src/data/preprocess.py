import re
import pandas as pd


def clean_text(text: str) -> str:
    """Normalize text while retaining letters, digits, plus signs, and # in skills."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_jobs(df: pd.DataFrame) -> pd.DataFrame:
    required = {"title", "skills", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Job data is missing required columns: {', '.join(sorted(missing))}")
    out = df.copy().fillna("")
    out["search_text"] = (out["title"].astype(str) + " " + out["skills"].astype(str) + " " + out["description"].astype(str)).map(clean_text)
    return out
