import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    attn = cache['attn']
    sum_dot = np.sum(d_attn * attn, axis=-1, keepdims=True)
    d_scores = attn * (d_attn - sum_dot)
    return d_scores
