def lower_bounded_decay(z, A, g_min=-5.0):
    """alpha = exp(g_min * sigmoid(exp(A) * z)), each entry in [exp(g_min), 1).

    z: (T, dk) decay logits.  A: scalar per-head log-scale.
    """
    log_scale = np.exp(A)
    g = g_min * (1 / (1 + np.exp(-log_scale * z)))
    return np.exp(g)
