import numpy as np

def fake_quantize_symmetric(x: np.ndarray, scale: float, bits: int = 8) -> np.ndarray:
    """qmax=2^(bits-1)-1; return clamp(round(x/scale),-qmax,qmax)*scale"""
    qmax = 2 ** (bits - 1) - 1
    x_s = np.round(x / scale)
    return np.clip(x_s, -qmax, qmax) * scale

def fake_quantize_ste_grad(x: np.ndarray, scale: float, bits: int = 8) -> np.ndarray:
    """STE mask: 1 where |x/scale|<=qmax else 0."""
    qmax = 2 ** (bits - 1) - 1
    x_s = np.abs(x / scale)
    return np.where(x_s <= qmax, 1.0, 0.0)
