import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    weights = cache['weights']
    dx = d_out @ weights.T
    return dx
