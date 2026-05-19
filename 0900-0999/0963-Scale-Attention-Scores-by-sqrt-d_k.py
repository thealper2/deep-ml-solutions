import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    n_q, d_k = K.shape
    raw_attn = Q @ K.T
    scale = np.sqrt(d_k)
    raw_attn_scaled = raw_attn / scale
    e_x = np.exp(raw_attn_scaled - np.max(raw_attn_scaled, axis=-1, keepdims=True))
    softmax_scores = e_x / e_x.sum(axis=-1, keepdims=True)
    softmax_scores = np.round(softmax_scores, 4)
    return softmax_scores.tolist()