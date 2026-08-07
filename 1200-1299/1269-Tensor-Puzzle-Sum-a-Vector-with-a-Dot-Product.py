import numpy as np

def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""
    ones = np.ones_like(a)
    return np.dot(a, ones)
