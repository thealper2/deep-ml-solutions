import numpy as np

def compute_fvd(features_real, features_gen) -> float:
    """
    Compute the Frechet Video Distance between two sets of feature vectors.

    Args:
        features_real: array-like of shape (N, D) -- features of real clips.
        features_gen:  array-like of shape (M, D) -- features of generated clips.

    Returns:
        float: FVD score (>= 0).
    """
    features_real = np.array(features_real)
    features_gen = np.array(features_gen)
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    mu_fake = np.mean(features_gen, axis=0)
    sigma_fake = np.cov(features_gen, rowvar=False)

    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    if sigma_fake.ndim == 0:
        sigma_fake = np.array([[sigma_fake]])

    mean_diff = mu_real - mu_fake
    mean_dist = np.sum(mean_diff ** 2)
    cov_product = np.dot(sigma_real, sigma_fake)
    eigenvalues = np.linalg.eigvals(cov_product)
    eigenvalues = np.maximum(eigenvalues.real, 0)
    trace_sqrt = np.sum(np.sqrt(eigenvalues))
    trace_real = np.trace(sigma_real)
    trace_fake = np.trace(sigma_fake)
    fvd = mean_dist + trace_real + trace_fake - 2 * trace_sqrt
    fvd = max(fvd, 0)
    return float(fvd)