def compute_delta(x, weight, bias=None):
    """Compute a strictly positive per-token timestep Delta.

    x: (B, L, E), weight: (E, E) nn.Linear layout, bias: optional (E,).
    Returns delta of shape (B, L, E).
    """
    out = x @ weight.T

    if bias is not None:
        out = out + bias

    delta = F.softplus(out)
    return delta