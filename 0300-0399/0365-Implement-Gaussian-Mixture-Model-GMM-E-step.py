import numpy as np

def gmm_e_step(X: np.ndarray, means: np.ndarray, variances: np.ndarray, mixing_coeffs: np.ndarray) -> np.ndarray:
    """
    Compute the E-step of Gaussian Mixture Model.
    
    Args:
        X: Data points of shape (n_samples,)
        means: Component means of shape (n_components,)
        variances: Component variances of shape (n_components,)
        mixing_coeffs: Mixing coefficients of shape (n_components,)
    
    Returns:
        Responsibility matrix of shape (n_samples, n_components)
    """
    n_samples = len(X)
    n_components = len(means)
    weighted_likelihoods = np.zeros((n_samples, n_components))

    for k in range(n_components):
        pdf = (1 / np.sqrt(2 * np.pi * variances[k])) * np.exp(-(X - means[k]) ** 2 / (2 * variances[k]))
        weighted_likelihoods[:, k] = mixing_coeffs[k] * pdf

    row_sums = np.sum(weighted_likelihoods, axis=1, keepdims=True)
    responsibilities = weighted_likelihoods / row_sums
    return responsibilities
