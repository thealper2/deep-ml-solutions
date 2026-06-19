def init_linear_layer(in_features, out_features, seed=0):
    W = he_init((in_features, out_features), in_features, seed)
    b = init_zero_bias(out_features)
    return {'W': W, 'b': b}
