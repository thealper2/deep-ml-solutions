import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape
    i = np.arange(d // 2)
    freqs = base ** (-2 * i / d)
    angles = np.outer(positions, freqs)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    x_even_new = x_even * cos_vals - x_odd * sin_vals
    x_odd_new = x_even * sin_vals + x_odd * cos_vals
    result = np.zeros_like(x)
    result[:, 0::2] = x_even_new
    result[:, 1::2] = x_odd_new
    return result