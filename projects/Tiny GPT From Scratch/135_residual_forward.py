def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    out = x + sublayer_out
    return out
