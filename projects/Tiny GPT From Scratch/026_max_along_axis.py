import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis=axis)
