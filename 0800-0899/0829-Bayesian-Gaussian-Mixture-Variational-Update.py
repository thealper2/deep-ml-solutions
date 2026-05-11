import numpy as np

def bayesian_gmm_update_means(X: np.ndarray, r: np.ndarray, beta0: float, m0: np.ndarray) -> np.ndarray:
    """Compute posterior component means for a Bayesian GMM variational update.

    Returns a (K, D) numpy array of updated means.
    """
    N, D = X.shape
    K = r.shape[1]
    Nk = np.sum(r, axis=0)
    weighted_sum = r.T @ X
    beta_k = beta0 + Nk
    m_k = (beta0 * m0 + weighted_sum) / beta_k[:, np.newaxis]
    m_k[Nk == 0] = m0
    return m_k