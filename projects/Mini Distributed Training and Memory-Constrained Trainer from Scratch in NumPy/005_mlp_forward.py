def mlp_forward(x, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    z1 = linear_forward(x, W1, b1)
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, W2, b2)
    cache = {'a1': a1, 'x': x, 'z1': z1, 'z2': z2}
    return z2, cache
