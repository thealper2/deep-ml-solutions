import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return ~mask
    
