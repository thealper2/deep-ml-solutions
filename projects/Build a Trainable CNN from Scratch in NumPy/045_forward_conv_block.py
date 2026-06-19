def forward_conv_block(x, W, b, pool_size, stride, pad):
    conv_out, conv_cache = conv2d_forward(x, W, b, stride, pad)
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = maxpool2d_forward(relu_out, pool_size, pool_size)
    cache = {
        'conv_cache': conv_cache,
        'relu_cache': relu_cache,
        'pool_cache': pool_cache,
    }
    return pool_out, cache
