import numpy as np

def bucketize(v: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return bucket indices for v given sorted boundaries."""
    return (v[:, None] >= boundaries).sum(axis=1).astype(int)
