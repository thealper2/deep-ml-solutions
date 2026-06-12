def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    row_sums = row_sums_of_counts(n_matrix)
    return n_matrix / row_sums
