import numpy as np

def marginal_belief_revision(B, B0):
    """ΔB_k = B_k - B_{k-1} with B_0 = B0."""
    B = np.array(B)
    B_extended = np.concatenate([[B0], B])
    return np.diff(B_extended)    
