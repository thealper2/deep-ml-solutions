import numpy as np

NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)

def nf4_quantize(x: np.ndarray):
    """
    NF4 block quant. scale=absmax; y=x/scale; q=argmin |y-NF4|; x_hat=scale*NF4[q].
    If absmax==0: q all 7, scale 1.0, x_hat 0.
    Ties: smallest index (np.argmin).
    Returns q (int), scale (float), x_hat (float array)
    """
    scale = np.max(np.abs(x))
    if scale == 0:
        q = np.full(x.shape, 7, dtype=np.uint8)
        scale = 1.0
        x_hat = np.zeros_like(x)
        return q, scale, x_hat

    y = x / scale
    distances = np.abs(y[:, None] - NF4[None, :])
    q = np.argmin(distances, axis=1)
    x_hat = scale * NF4[q]
    return q, scale, x_hat
