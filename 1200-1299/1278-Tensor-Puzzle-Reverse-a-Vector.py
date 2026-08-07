import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    n = a.shape[0]
    out = np.zeros_like(a)
    for i in range(n):
        out[i] = a[n - i - 1]

    return out
