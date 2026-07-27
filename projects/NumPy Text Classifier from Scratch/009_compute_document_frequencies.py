def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    return np.sum(bow_matrix > 0, axis=0)
