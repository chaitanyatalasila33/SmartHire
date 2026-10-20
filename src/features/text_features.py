from sklearn.feature_extraction.text import TfidfVectorizer


def make_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000, sublinear_tf=True)
