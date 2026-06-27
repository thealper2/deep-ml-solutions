import numpy as np

def layer_norm_apply(x, ln_params, eps=1e-5):
    """Normalize x over its last axis and apply gamma, beta."""
    mean, variance = compute_mean_variance(x, eps)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return ln_params['gamma'] * x_norm + ln_params['beta']
