import numpy as np

def make_empty_board():
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    return np.zeros((6, 7), dtype=int)
