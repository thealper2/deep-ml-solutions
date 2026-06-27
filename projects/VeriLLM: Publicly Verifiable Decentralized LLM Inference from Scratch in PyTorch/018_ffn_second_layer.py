def ffn_second_layer(h, ffn_params):
    z = linear_projection(h, ffn_params['W2'], ffn_params.get('b2', None))
    return z
