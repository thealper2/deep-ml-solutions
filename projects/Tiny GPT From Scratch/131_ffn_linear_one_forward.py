def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    linear = linear_forward(x, w1)
    with_bias = bias_add_forward(linear['y'], b1)
    cache = {'x': x, 'w1': w1}
    return {'h1': with_bias['y'], 'cache': cache}
