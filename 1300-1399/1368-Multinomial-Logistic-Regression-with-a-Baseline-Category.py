import numpy as np


def multinomial_probs(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Class probabilities under a baseline-category multinomial logistic model.

    Args:
        x (np.ndarray): (p,) feature vector, without an intercept term.
        B (np.ndarray): (K-1, p+1) coefficients for classes 1..K-1, intercept first.

    Returns:
        np.ndarray: (K,) probabilities, class 0 first, summing to 1.
    """
    d = np.concatenate([[1.0], x])
    eta = np.concatenate([[0.0], B @ d])
    eta_stable = eta - np.max(eta)
    exp_eta = np.exp(eta_stable)
    probs = exp_eta / np.sum(exp_eta)
    return probs