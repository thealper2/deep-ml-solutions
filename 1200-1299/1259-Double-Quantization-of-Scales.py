import numpy as np

def double_quantize_scales(scales: np.ndarray):
    """
    Symmetric INT8 absmax quant on a vector of scales.
    scale2 = max|scales|/127 (or 1.0); q = clamp(round(scales/scale2),-127,127)
    Returns q_scales, scale2, scales_hat
    """
    scale2 = np.max(np.abs(scales)) / 127 if np.any(scales) else 1.0
    q = np.clip(np.round(scales / scale2), -127, 127)
    scales_hat = q * scale2
    return q.astype(np.uint8), scale2, scales_hat
