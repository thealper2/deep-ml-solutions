import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    return x.T @ dout
