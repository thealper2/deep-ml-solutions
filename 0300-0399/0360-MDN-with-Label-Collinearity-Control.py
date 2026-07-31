import numpy as np

def mdn_with_collinearity(f: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          sigma_tilde_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply MDN with label collinearity control.
    
    Args:
        f: Features, shape (M, D)
        X: Metadata, shape (M, K)
        y: Labels, shape (M,) or (M, 1)
        sigma_tilde_inv: Inverse covariance of augmented [X, y] matrix, shape (K+1, K+1)
        N: Total training samples
    
    Returns:
        Features with metadata (but not label) effects removed
    """
    batch_size = f.shape[0]
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    X_tilde = np.hstack([X, y])
    beta = sigma_tilde_inv @ (X_tilde.T @ f)
    d_metadata = X.shape[1]
    beta_meta = beta[:d_metadata]
    f_meta_pred = X @ beta_meta
    f_residual = f - f_meta_pred
    return f_residual
