def recompute_block_activations(x, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    z1 = linear_forward(x, W1, b1)
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, W2, b2)
    
    return {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}
