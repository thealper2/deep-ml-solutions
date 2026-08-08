import numpy as np

def asymmetric_quantize(x: np.ndarray, bits: int = 8):
    """
    Asymmetric uniform quantization.

    Returns q (int array), scale (float), zero_point (int), x_hat (float array).
    Unsigned grid: qmin=0, qmax=2^bits-1.
    """
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max - x_min < 1e-12:
        q = np.zeros_like(x, dtype=np.int32)
        scale = 1.0
        zero_point = 0
        x_hat = x.copy()
        return q, scale, zero_point, x_hat

    qmin = 0
    qmax = 2 ** bits - 1
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = np.clip(np.round(qmin - x_min / scale), qmin, qmax)
    q = np.clip(np.round(x / scale + zero_point), qmin, qmax)
    x_hat = scale * (q - zero_point)
    return q.astype(np.int32), scale, int(zero_point), x_hat

