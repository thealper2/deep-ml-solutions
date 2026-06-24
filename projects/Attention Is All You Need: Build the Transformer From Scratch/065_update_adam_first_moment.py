import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    m_t = beta1 * m_prev + (1 - beta1) * grad
    return m_t
