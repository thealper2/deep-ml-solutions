def ffn_first_layer_gelu(x, ffn_params):
    z = linear_projection(x, ffn_params['W1'], ffn_params.get('b1', None))
    constants = np.sqrt(2.0 / np.pi)
    return 0.5 * z * (1.0 + np.tanh(constants * (z + 0.044715 * np.power(z, 3))))
