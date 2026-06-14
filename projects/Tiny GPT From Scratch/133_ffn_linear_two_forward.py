def ffn_linear_two_forward(a1, w2, b2):
    linear = linear_forward(a1, w2)
    with_bias = bias_add_forward(linear['y'], b2)
    cache = {'a1': a1, 'w2': w2}
    return {'h2': with_bias['y'], 'cache': cache}
