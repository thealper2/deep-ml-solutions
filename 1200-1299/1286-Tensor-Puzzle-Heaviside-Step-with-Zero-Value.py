import numpy as np

def heaviside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise heaviside with zero-value b."""
    neg_mask = (a < 0)
    zero_mask = (a == 0)
    pos_mask = (a > 0)
    return neg_mask * 0.0 + zero_mask * b + pos_mask * 1.0
