import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    return np.sum(matrix, axis=1, keepdims=True)
