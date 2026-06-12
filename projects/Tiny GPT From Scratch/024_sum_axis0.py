import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, axis=0)
