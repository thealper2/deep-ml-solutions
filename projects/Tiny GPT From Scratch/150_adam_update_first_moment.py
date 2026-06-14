import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    return beta1 * m + (1 - beta1) * grad
