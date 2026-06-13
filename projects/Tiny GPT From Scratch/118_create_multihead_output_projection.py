def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    Wo = make_2d_random(d_model, d_model, seed=0)
    Wo = scale_w_small(Wo, scale)
    return Wo
