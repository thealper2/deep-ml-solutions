import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    N, C, H, W = x.shape
    H_out = (H - kernel_size + stride - 1) // stride + 1
    W_out = (W - kernel_size + stride - 1) // stride + 1
    pad_h = max(0, (H_out - 1) * stride + kernel_size - H)
    pad_w = max(0, (W_out - 1) * stride + kernel_size - W)
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        H, W = x.shape[2], x.shape[3]

    windows = sliding_window_view(x, window_shape=(kernel_size, kernel_size), axis=(2, 3))
    strided_windows = windows[:, :, ::stride, ::stride, :, :]
    pooled = np.max(strided_windows, axis=(-2, -1))
    return pooled