import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    x_max = np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
