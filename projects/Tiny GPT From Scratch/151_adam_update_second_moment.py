def adam_update_second_moment(v_prev, grad, beta2):
    """Update Adam's second-moment estimate v using squared gradient EMA."""
    v_t = beta2 * v_prev + (1 - beta2) * (grad ** 2)
    return v_t
