def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    return bow_matrix * idf
