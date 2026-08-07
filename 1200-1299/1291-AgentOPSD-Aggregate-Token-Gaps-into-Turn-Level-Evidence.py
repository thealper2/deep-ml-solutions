import numpy as np

def turn_evidence(token_gaps, mask=None):
    """Sum token-level δ into per-turn evidence e_k."""
    if mask is None:
        if token_gaps.ndim == 1:
            return np.sum(token_gaps)
        else:
            return np.sum(token_gaps, axis=-1)
    else:
        return np.sum(token_gaps * mask, axis=-1 if token_gaps.ndim > 1 else None)
