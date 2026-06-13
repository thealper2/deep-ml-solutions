def create_qkv_projections(d_model, d_head, scale=0.02):
    Wq = make_2d_random(d_model, d_head, seed=0)
    Wk = make_2d_random(d_model, d_head, seed=1)
    Wv = make_2d_random(d_model, d_head, seed=2)

    Wq = scale_w_small(Wq, scale)
    Wk = scale_w_small(Wk, scale)
    Wv = scale_w_small(Wv, scale)

    return {
        'Wq': Wq,
        'Wk': Wk,
        'Wv': Wv,
    }
