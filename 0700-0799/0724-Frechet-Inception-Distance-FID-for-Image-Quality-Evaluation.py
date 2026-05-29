import numpy as np

def fid_score(real_features, generated_features) -> float:
    """
    Compute the Frechet Inception Distance between two feature sets.

    Args:
        real_features: array-like of shape (N1, D)
        generated_features: array-like of shape (N2, D)

    Returns:
        float: the FID value.
    """
    real_features = np.array(real_features)
    generated_features = np.array(generated_features)

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)

    if real_features.shape[1] == 1:
        sigma_real = np.array([[sigma_real]])
        sigma_gen = np.array([[sigma_gen]])

    ssdiff = np.sum((mu_real - mu_gen) ** 2)
    cov_prod = np.dot(sigma_real, sigma_gen)

    evals, evecs = np.linalg.eigh(cov_prod)

    covmean = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T

    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return fid