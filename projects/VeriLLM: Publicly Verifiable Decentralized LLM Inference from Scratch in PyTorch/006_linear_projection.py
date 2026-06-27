import numpy as np

def linear_projection(x, weight, bias=None):
    """Affine map y = x @ weight + bias used throughout the transformer."""
    z = x @ weight
    if bias is not None:
        z = z + bias

    return z
