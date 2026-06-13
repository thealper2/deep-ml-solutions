import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    return x.shape[1]
