import numpy as np

def mdn_layer(f: np.ndarray, X: np.ndarray, sigma_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply Metadata Normalization to features.
    
    Args:
        f: Features array of shape (M, D) where M is batch size, D is feature dimension
        X: Metadata matrix of shape (M, K) where K is number of metadata variables
        sigma_inv: Pre-computed inverse covariance matrix of shape (K, K)
        N: Total number of training samples
    
    Returns:
        Residualized features of shape (M, D), orthogonal to metadata subspace
    """
    batch_size = f.shape[0]
    beta = (N / batch_size) * sigma_inv @ (X.T @ f)
    f_residual = f - X @ beta
    return f_residual
