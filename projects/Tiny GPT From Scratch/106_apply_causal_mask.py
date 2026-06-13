import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    masked_scores = np.where(causal_mask, scaled_scores, -np.inf)
    return masked_scores
