import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input.
    """
    N, C, H, W = x.shape
    y = np.zeros_like(x)
    pad = (n - 1) // 2
    padded_x = np.pad(x, ((0, 0), (pad, pad), (0, 0), (0, 0)), mode='constant')

    for i in range(C):
        start = i
        end = i + n
        channel_sum_sq = np.sum(padded_x[:, start:end, :, :] ** 2, axis=1)
        y[:, i, :, :] = x[:, i, :, :] / ((k + alpha * channel_sum_sq) ** beta)

    return y