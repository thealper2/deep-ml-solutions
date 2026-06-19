def linear_backward(dout, cache):
    x = cache['x']
    weights = cache['weights']

    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(x, dout)
    db = linear_grad_bias(dout)

    return dx, dW, db
