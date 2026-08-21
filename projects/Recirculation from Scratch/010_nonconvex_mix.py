def nonconvex_mix(s, d, alpha):
    """Nonconvex mix: destination plus a scaled matched source."""
    matched_s = match_source_norm(s, d)
    return d + alpha * matched_s