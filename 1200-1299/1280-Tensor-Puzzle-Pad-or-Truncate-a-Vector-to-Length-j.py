import numpy as np

def pad_to(a: np.ndarray, j: int) -> np.ndarray:
    """Pad a with zeros (or truncate) to length j."""
    out = np.zeros(j, dtype=a.dtype)
    copy_len = min(len(a), j)
    out[:copy_len] = a[:copy_len]
    return out
