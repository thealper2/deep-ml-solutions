import numpy as np

def sequence_mask(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Zero out positions j >= lengths[i] in each row i."""
    seq_len = values.shape[1]
    positions = np.arange(seq_len)
    mask = positions < lengths[:, None]
    return values * mask.astype(values.dtype)
