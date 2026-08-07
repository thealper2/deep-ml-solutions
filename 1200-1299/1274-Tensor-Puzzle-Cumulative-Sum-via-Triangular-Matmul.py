import numpy as np

def cumsum(a: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of 1-D array a without np.cumsum."""
    n = len(a)
    i = np.arange(n)
    j = np.arange(n)
    tril_matrix = (i[:, None] >= j).astype(a.dtype)
    return tril_matrix @ a
