def lm_head_logits(hidden, lm_head_params):
    linear = linear_projection(
        hidden, 
        lm_head_params['W'],
        lm_head_params['b'],
    )
    return linear
