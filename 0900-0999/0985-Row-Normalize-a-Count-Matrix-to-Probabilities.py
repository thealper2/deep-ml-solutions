import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    row_sums = np.sum(counts, axis=1)
    normalized = counts / row_sums[:, np.newaxis]
    normalized[np.isnan(normalized)] = 0.0
    return normalized.tolist()