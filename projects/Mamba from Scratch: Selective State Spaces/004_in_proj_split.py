def in_proj_split(u, weight, bias=None):
    """Project tokens to expanded inner width and split into SSM input x and gate z."""
    out = u @ weight.T

    if bias is not None:
        out = out + bias

    E = out.shape[-1] // 2
    x = out[..., :E]
    z = out[..., E:]
    return x, zdef in_proj_split(u, weight, bias=None):
    """Project tokens to expanded inner width and split into SSM input x and gate z."""
    out = u @ weight.T

    if bias is not None:
        out = out + bias

    E = out.shape[-1] // 2
    x = out[..., :E]
    z = out[..., E:]
    return x, z