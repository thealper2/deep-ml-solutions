def backward_conv_block(dout, cache):
    pool_cache = cache['pool_cache']
    relu_cache = cache['relu_cache']
    conv_cache = cache['conv_cache']

    d_relu_out = maxpool2d_backward(dout, pool_cache)
    d_conv_out = relu_backward(d_relu_out, relu_cache)
    dx, dW, db = conv2d_backward(d_conv_out, conv_cache)

    return dx, dW, db
