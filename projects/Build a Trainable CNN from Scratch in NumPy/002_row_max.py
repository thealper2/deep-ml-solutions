import numpy as np

def row_max(matrix):
    return np.max(matrix, axis=1, keepdims=True)
