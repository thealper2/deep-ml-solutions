def hadamard_mix(s, d, alpha, beta):
    """Hadamard mix of matched source and destination."""
    matched_s = match_source_norm(s, d)
    return alpha * matched_s + beta * d