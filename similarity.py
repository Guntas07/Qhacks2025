try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except Exception:
    # Allow callers to detect unavailability and choose a fallback
    _SKLEARN_AVAILABLE = False


def sklearn_available() -> bool:
    return _SKLEARN_AVAILABLE


def rank_by_tfidf_cosine(query_text: str, product_texts: list[str]) -> list[float]:
    """
    Compute TF-IDF vectors for the query and product texts, then return cosine similarity scores.

    Args:
        query_text: Preprocessed query terms as a single string.
        product_texts: List of preprocessed product texts (name + description).

    Returns:
        List of cosine similarity scores aligned with product_texts in order.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not available for TF-IDF similarity")

    # First document is the query; remaining are products
    corpus = [query_text] + product_texts
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    # Compute cosine similarity of query (row 0) vs each product (rows 1..N)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).ravel()
    return sims.tolist()

