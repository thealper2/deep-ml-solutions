import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    scores = np.matmul(q, k.transpose(0, 2, 1))
    return scores
