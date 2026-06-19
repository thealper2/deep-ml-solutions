def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    fan_in = in_channels * kernel_size * kernel_size
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    W = he_init(shape, fan_in, seed)
    b = init_zero_bias(out_channels)
    return {'W': W, 'b': b}
