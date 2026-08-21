def concat_residuals(s, d):
    """Concatenate source and destination residuals along the last axis."""
    return torch.cat([s, d], dim=-1)