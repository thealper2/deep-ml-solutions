import numpy as np

def reshape_turn_advantages(A_seq, w, lam: float = 0.5):
    """Ã_k = A_seq * ((1-lam) + lam * w_k)."""
    return np.array(A_seq) * ((1 - np.array(lam)) + np.array(lam) * np.array(w))
