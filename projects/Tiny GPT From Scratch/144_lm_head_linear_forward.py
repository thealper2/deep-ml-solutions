def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B,T,d_model) to logits (B,T,vocab_size)."""
    linear = linear_forward(x, w_lm)
    with_bias = bias_add_forward(linear['y'], b_lm)
    return {'logits': with_bias['y'], 'cache': {'x': x, 'w_lm': w_lm}}
