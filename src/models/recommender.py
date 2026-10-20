import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data.preprocess import validate_jobs
from src.features.text_features import make_tfidf_vectorizer


def recommend_jobs(resume_text: str, jobs: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Rank jobs by cosine similarity between resume and job text."""
    if not str(resume_text or "").strip():
        return pd.DataFrame(columns=list(jobs.columns) + ["match_score"])
    prepared = validate_jobs(jobs)
    vectorizer = make_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(prepared["search_text"].tolist() + [resume_text])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    prepared["match_score"] = (scores * 100).round(1)
    prepared = prepared.sort_values("match_score", ascending=False).head(max(1, int(top_n)))
    return prepared.drop(columns=["search_text"], errors="ignore").reset_index(drop=True)
