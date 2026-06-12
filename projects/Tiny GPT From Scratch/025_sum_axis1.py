import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    return np.sum(arr, axis=1)
