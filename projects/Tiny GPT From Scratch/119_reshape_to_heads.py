import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    B, T, d_model = x.shape
    return x.reshape(B, T, n_heads, d_head)
