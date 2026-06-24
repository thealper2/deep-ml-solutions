import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    v_t = beta2 * v_prev + (1 - beta2) * grad ** 2
    return v_t
