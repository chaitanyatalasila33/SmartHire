from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from src.features.text_features import make_tfidf_vectorizer


def cluster_job_descriptions(descriptions: list[str], n_clusters: int = 3, random_state: int = 42):
    """Optional exploratory clustering helper; returns labels and a 2D projection when possible."""
    if len(descriptions) < 2:
        raise ValueError("At least two job descriptions are required for clustering.")
    k = max(2, min(int(n_clusters), len(descriptions)))
    vectorizer = make_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(descriptions)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(matrix)
    if matrix.shape[1] >= 2 and matrix.shape[0] >= 2:
        coords = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(matrix)
    else:
        coords = None
    return labels, coords
