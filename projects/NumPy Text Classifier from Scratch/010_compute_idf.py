def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    return np.log((n_docs + 1) / (df + 1)) + 1
