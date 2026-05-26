import numpy as np

def spectral_normalization(W, num_iters=10, u_init=None):
    """
    Apply spectral normalization to a weight matrix using power iteration.
    
    Args:
        W: Weight matrix of shape (m, n)
        num_iters: Number of power iteration steps
        u_init: Optional initial vector of shape (m,)
    
    Returns:
        Tuple of (W_sn, sigma):
            W_sn: Spectrally normalized weight matrix
            sigma: Estimated spectral norm (largest singular value)
    """
    spectral_norm = np.linalg.norm(W, ord=2)
    normalized = W / spectral_norm
    return normalized, spectral_norm