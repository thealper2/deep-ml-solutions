import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    m_hat = m_t / (1 - beta1 ** step)
    v_hat = v_t / (1 - beta2 ** step)
    return m_hat, v_hat
