import numpy as np

def partial_rope(x, partial_rotary_factor, theta_base=10000.0, offset=0):
    """
    Apply partial Rotary Position Embedding to x.

    Args:
        x: np.ndarray of shape (batch, num_heads, seq_len, head_dim)
        partial_rotary_factor: float in (0, 1]
        theta_base: float, base of the inverse-frequency schedule
        offset: int, starting absolute position

    Returns:
        np.ndarray of the same shape as x
    """
    batch, num_heads, seq_len, head_dim = x.shape
    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
    i = np.arange(rotary_dim // 2, dtype=np.float64)
    inv_freq = 1.0 / (theta_base ** (2 * i / rotary_dim))
    positions = np.arange(offset, offset + seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)
    angles_dup = np.concatenate([angles, angles], axis=-1)
    cos_vals = np.cos(angles_dup)
    sin_vals = np.sin(angles_dup)
    cos_vals = cos_vals.reshape(1, 1, seq_len, rotary_dim)
    sin_vals = sin_vals.reshape(1, 1, seq_len, rotary_dim)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    x_rot_out = x_rot * cos_vals + rotated * sin_vals
    result = np.concatenate([x_rot_out, x_pass], axis=-1)
    return result
