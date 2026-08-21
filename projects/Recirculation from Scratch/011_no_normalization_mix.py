def no_normalization_mix(s, d, alpha):
    """Mix source into destination with no renormalization using the raw source."""
    return (1 - alpha) * d + alpha * s