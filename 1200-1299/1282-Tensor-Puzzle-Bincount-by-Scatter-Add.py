import numpy as np

def bincount(a: np.ndarray, n_bins: int) -> np.ndarray:
    """Count occurrences of 0..n_bins-1 in integer array a."""
    bins = np.arange(n_bins)
    one_hot = (a[:, None] == bins)
    return one_hot.sum(axis=0).astype(int)
