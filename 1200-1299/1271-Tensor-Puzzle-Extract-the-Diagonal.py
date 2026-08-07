import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    n = A.shape[0]
    d = [A[i, i] for i in range(n)]
    return np.array(d, dtype=np.float64)
