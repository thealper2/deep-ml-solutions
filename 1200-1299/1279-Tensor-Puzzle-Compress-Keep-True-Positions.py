import numpy as np

def compress(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pack v[g] into the front of a zero vector of length len(v)."""
    n = len(g)
    out = np.zeros(n, dtype=v.dtype)
    true_indices = np.arange(n)[g]
    k = len(true_indices)
    out[:k] = v[true_indices]
    return out
