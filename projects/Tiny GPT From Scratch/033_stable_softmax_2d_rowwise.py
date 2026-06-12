import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    x_max = np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
