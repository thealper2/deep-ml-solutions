def mlp_backward(dy_pred, cache, params):
    x = cache['x']
    z1 = cache['z1']
    a1 = cache['a1']
    W1 = params['W1']
    W2 = params['W2']

    d_a1, dW2, db2 = linear_backward(dy_pred, a1, W2)
    d_z1 = relu_backward(d_a1, z1)

    dx, dW1, db1 = first_linear_backward(d_z1, x, W1)

    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2,
    }
