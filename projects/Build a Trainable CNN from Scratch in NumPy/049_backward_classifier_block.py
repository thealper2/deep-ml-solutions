def backward_classifier_block(dlogits, cache):
    fc2_cache = cache['fc2_cache']
    relu_cache = cache['relu_cache']
    fc1_cache = cache['fc1_cache']
    flatten_cache = cache['flatten_cache']

    d_relu_out, dW2, db2 = linear_backward(dlogits, fc2_cache)
    d_fc1_out = relu_backward(d_relu_out, relu_cache)
    d_flat_out, dW1, db1 = linear_backward(d_fc1_out, fc1_cache)
    dx = flatten_backward(d_flat_out, flatten_cache)

    return {
        'dx': dx,
        'fc1': {'dW': dW1, 'db': db1},
        'fc2': {'dW': dW2, 'db': db2},   
    }
