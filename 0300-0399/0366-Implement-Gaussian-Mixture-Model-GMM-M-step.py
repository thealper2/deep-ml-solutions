import numpy as np

def gmm_m_step(X, gamma):
    """
    Perform the M-step of the EM algorithm for Gaussian Mixture Model.
    
    Args:
        X: Data points, shape (N, D)
        gamma: Responsibilities from E-step, shape (N, K)
    
    Returns:
        tuple: (means, covariances, mixing_coeffs)
            - means: Updated component means, shape (K, D)
            - covariances: Updated component covariances, shape (K, D, D)
            - mixing_coeffs: Updated mixing coefficients, shape (K,)
    """
    X = np.array(X)
    gamma = np.array(gamma)
    
    N, D = X.shape
    K = gamma.shape[1]
    
    N_k = np.sum(gamma, axis=0)
    
    means = np.zeros((K, D))
    for k in range(K):
        means[k] = np.sum(gamma[:, k:k+1] * X, axis=0) / N_k[k]
    
    covariances = np.zeros((K, D, D))
    for k in range(K):
        deviations = X - means[k]
        weighted_cov = np.zeros((D, D))
        for n in range(N):
            weighted_cov += gamma[n, k] * np.outer(deviations[n], deviations[n])

        covariances[k] = weighted_cov / N_k[k]
    
    mixing_coeffs = N_k / N
    return means, covariances, mixing_coeffs
