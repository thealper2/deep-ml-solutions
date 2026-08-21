def ramped_alpha(t, alpha, ramp_steps=10):
    """Compute the ramped mixture coefficient for a 0-indexed token position t."""
    return alpha * (t / ramp_steps) if t < ramp_steps else alpha