import numpy as np

def rescale_update(O_prime, gamma):
    """
    Rescale an orthogonalized update matrix to match AdamW-style RMS.
    Args:
        O_prime: 2D numpy array of shape (n, m)
        gamma: float scaling factor
    Returns:
        Rescaled update as a nested list.
    """
    n, m = O_prime.shape
    scale = np.sqrt(max(n, m)) * gamma
    scaled = O_prime * scale
    return scaled.tolist()