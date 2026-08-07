import numpy as np

def flatten(A: np.ndarray) -> np.ndarray:
    """Row-major flatten of 2-D array A without reshape/ravel."""
    rows, cols = A.shape
    n = rows * cols
    linear_indices = np.arange(n)
    row_indices = linear_indices // cols
    col_indices = linear_indices % cols
    return A[row_indices, col_indices]
