import numpy as np

def outcome_aligned_credit(delta_B, A_seq):
    """q_k = sign(A_seq) * ΔB_k."""
    return np.sign(A_seq) * np.array(delta_B)
