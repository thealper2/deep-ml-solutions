import numpy as np

def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = 1.0 if i <= j else 0.0
    
    return out
