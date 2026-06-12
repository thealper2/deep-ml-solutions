def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    return np.sum(n_matrix, axis=1, keepdims=True)
