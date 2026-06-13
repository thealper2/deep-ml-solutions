import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    return np.matmul(weights, v_heads)
