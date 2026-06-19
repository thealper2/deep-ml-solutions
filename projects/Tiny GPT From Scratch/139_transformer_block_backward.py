def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block."""
    x = cache["attn_branch"]["x"]
    cache = _complete_block_cache(x, block_params)

    attn_branch = cache["attn_branch"]
    ffn_branch = cache["ffn_branch"]

    d_ln2_out, ffn_grads = _ffn_sublayer_backward(
        d_y, ffn_branch["sublayer_cache"], block_params["ffn"]
    )
    d_h1_sub, d_g2, d_b2 = layernorm_backward_affine(
        d_ln2_out, ffn_branch["ln_cache"]
    )
    d_h1 = d_h1_sub + d_y

    d_ln1_out, attn_grads = _attn_sublayer_backward(
        d_h1, attn_branch["sublayer_cache"], block_params["attn"]
    )
    d_x_sub, d_g1, d_b1 = layernorm_backward_affine(
        d_ln1_out, attn_branch["ln_cache"]
    )
    d_x = d_x_sub + d_h1

    grads = {
        "ln1": {"gamma": d_g1, "beta": d_b1},
        "ln2": {"gamma": d_g2, "beta": d_b2},
        "attn": attn_grads,
        "ffn": ffn_grads,
    }
    return d_x, grads
