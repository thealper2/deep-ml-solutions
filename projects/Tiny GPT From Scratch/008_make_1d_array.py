import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    return np.array(values).flatten()
