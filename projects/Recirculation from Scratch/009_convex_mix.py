def convex_mix(s, d, alpha):
    """Convex mix of destination with a magnitude-matched source."""
    matched_s = match_source_norm(s, d)
    return (1 - alpha) * d + alpha * matched_s