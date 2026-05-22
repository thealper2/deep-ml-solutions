import numpy as np

def update_sliding_kv_cache(cache, new_kv, window: int):
    """
    Append new_kv to cache along the time axis and truncate to the last `window` tokens.

    Args:
        cache: np.ndarray of shape (B, H, T_old, D)
        new_kv: np.ndarray of shape (B, H, T_new, D)
        window: int, maximum number of tokens to retain

    Returns:
        np.ndarray of shape (B, H, min(T_old + T_new, window), D)
    """
    combined = np.concatenate((cache, new_kv), axis=2)
    if combined.shape[2] > window:
        combined = combined[:, :, -window:, :]

    return combined