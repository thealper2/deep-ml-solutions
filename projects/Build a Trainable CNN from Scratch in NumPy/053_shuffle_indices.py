import numpy as np

def shuffle_indices(n, seed=0):
    np.random.seed(seed)
    return np.random.permutation(n)
