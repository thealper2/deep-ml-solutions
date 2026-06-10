def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    return rng.random() < epsilon
