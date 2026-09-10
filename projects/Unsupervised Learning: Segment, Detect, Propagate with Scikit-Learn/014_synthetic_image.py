import numpy as np

def synthetic_image(size=48):
    img = np.zeros((size, size, 3), dtype=np.float64)
    half = size // 2

    g_left = np.linspace(0, 1, half)
    img[:, :half, 0] = 1.0
    img[:, :half, 1] = g_left[np.newaxis, :]
    img[:, :half, 2] = 0.0

    g_right = np.linspace(0, 1, size - half)
    img[:, half:, 0] = 0.0
    img[:, half:, 1] = g_right[np.newaxis, :]
    img[:, half:, 2] = 1.0

    q = size // 4
    img[:q, :q] = 1.0

    return img