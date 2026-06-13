import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    return scores / np.sqrt(d_head)
