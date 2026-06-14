def adam_bias_correction(m, v, beta1, beta2, t):
    """Return bias-corrected (m_hat, v_hat) for Adam at step t."""
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return m_hat, v_hat
