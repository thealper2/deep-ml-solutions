def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Args:
        x: ndarray of shape (B, T, d_model).
        block_params: dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        dict with 'y' (B, T, d_model) and 'cache' with keys
        'attn_branch' and 'ffn_branch'.
    """
    d_model = x.shape[-1]
    mean1 = x.mean(axis=-1, keepdims=True)
    var1 = x.var(axis=-1, keepdims=True)
    x_norm1 = (x - mean1) / np.sqrt(var1 + 1e-5)
    ln1_gamma = block_params["ln1"]["gamma"]
    ln1_beta = block_params["ln1"]["beta"]
    x_norm1 = x_norm1 * ln1_gamma + ln1_beta
    attn = block_params["attn"]
    Wv = attn.get("Wv", np.zeros((d_model, d_model)))
    Wo = attn.get("Wo", np.zeros((d_model, d_model)))
    bo = attn.get("bo", np.zeros(d_model))
    V = x_norm1 @ Wv
    attn_out = V @ Wo + bo
    h1 = x + attn_out
    mean2 = h1.mean(axis=-1, keepdims=True)
    var2 = h1.var(axis=-1, keepdims=True)
    x_norm2 = (h1 - mean2) / np.sqrt(var2 + 1e-5)
    ln2_gamma = block_params["ln2"]["gamma"]
    ln2_beta = block_params["ln2"]["beta"]
    x_norm2 = x_norm2 * ln2_gamma + ln2_beta
    ffn = block_params["ffn"]
    w1 = ffn.get("w1", np.zeros((d_model, d_model)))
    b1 = ffn.get("b1", np.zeros(d_model))
    w2 = ffn.get("w2", np.zeros((d_model, d_model)))
    b2 = ffn.get("b2", np.zeros(d_model))
    ffn_hidden = x_norm2 @ w1 + b1
    ffn_out = ffn_hidden @ w2 + b2
    y = h1 + ffn_out

    cache = {
        "attn_branch": {
            "x": x,
            "ln_cache": {"x": x, "mean": mean1, "var": var1},
            "sublayer_cache": {
                "x_norm": x_norm1,
                "attn_out": attn_out,
            },
        },
        "ffn_branch": {
            "x": h1,
            "ln_cache": {"x": h1, "mean": mean2, "var": var2},
            "sublayer_cache": {
                "x_norm": x_norm2,
                "ffn_hidden": ffn_hidden,
                "ffn_out": ffn_out,
            },
        },
    }

    return {"y": y, "cache": cache}
