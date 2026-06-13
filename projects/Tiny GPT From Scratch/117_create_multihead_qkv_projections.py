def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    Wq = scale_w_small(make_2d_random(d_model, d_model, seed=0), scale)
    Wk = scale_w_small(make_2d_random(d_model, d_model, seed=1), scale)
    Wv = scale_w_small(make_2d_random(d_model, d_model, seed=2), scale)
    
    return {'Wq': Wq, 'Wk': Wk, 'Wv': Wv}
