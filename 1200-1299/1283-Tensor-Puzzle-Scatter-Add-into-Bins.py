import numpy as np

def scatter_add(values: np.ndarray, indices: np.ndarray, n_bins: int) -> np.ndarray:
    """Sum values into bins given by indices."""
    one_hot = (indices[:, None] == np.arange(n_bins))
    weighted = one_hot * values[:, None]
    return weighted.sum(axis=0)
