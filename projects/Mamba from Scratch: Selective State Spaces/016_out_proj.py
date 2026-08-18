def out_proj(y, weight, bias=None):
    """Project gated scan output from d_inner back to d_model.

    y: (..., d_inner)
    weight: (d_model, d_inner)
    bias: (d_model,) or None
    Returns: (..., d_model)
    """
    out = y @ weight.T

    if bias is not None:
        out = out + bias

    return out