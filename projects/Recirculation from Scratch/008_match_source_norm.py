def match_source_norm(s, d):
    """Rescale s so its last-axis L2 matches d."""
    norm_s = last_axis_l2(s)
    norm_d = last_axis_l2(d)
    mask = norm_s > 0
    scale = torch.zeros_like(norm_s)
    scale[mask] = norm_d[mask] / norm_s[mask]
    return s * scale