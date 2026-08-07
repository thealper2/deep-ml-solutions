import numpy as np

def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""
    return np.array([1.0] * n, dtype=np.float64)
