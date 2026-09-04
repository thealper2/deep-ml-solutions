import numpy as np


def inverse_link(eta, family: str):
    """Map the linear predictor to the mean for a GLM family."""
    eta = np.asarray(eta, dtype=float)
    is_scalar = eta.ndim == 0

    if family == 'gaussian':
        result = eta
    elif family == 'binomial':
        result = 1.0 / (1.0 + np.exp(-eta))
    elif family == 'poisson':
        result = np.exp(eta)
    else:
        raise ValueError(f"Unknown family: {family}")

    if is_scalar and result.ndim == 0:
        return float(result)

    return result


def link(mu, family: str):
    """Map the mean to the linear predictor (the inverse of inverse_link)."""
    mu = np.asarray(mu, dtype=float)
    is_scalar = mu.ndim == 0

    if family == 'gaussian':
        result = mu
    elif family == 'binomial':
        result = np.log(mu / (1.0 - mu))
    elif family == 'poisson':
        result = np.log(mu)
    else:
        raise ValueError(f"Unknown family: {family}")

    if is_scalar and result.ndim == 0:
        return float(result)

    return result



def variance_function(mu, family: str):
    """Return V(mu), the mean-variance relationship for the family."""
    mu = np.asarray(mu, dtype=float)
    is_scalar = mu.ndim == 0

    if family == 'gaussian':
        result = np.ones_like(mu)
    elif family == 'binomial':
        result = mu * (1.0 - mu)
    elif family == 'poisson':
        result = mu
    else:
        raise ValueError(f"Unknown family: {family}")

    if is_scalar and result.ndim == 0:
        return float(result)

    return result