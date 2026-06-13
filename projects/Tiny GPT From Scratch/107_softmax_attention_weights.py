import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    max_x = np.max(masked_scores, axis=-1, keepdims=True)
    exp_x = np.exp(masked_scores - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
