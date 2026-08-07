import numpy as np

def vstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack 1-D arrays a and b as rows of a (2, n) matrix."""
    row_mask = np.arange(2).reshape(2, 1)
    return row_mask * b + (1 - row_mask) * a
