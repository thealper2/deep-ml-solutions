import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    x_max = np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
