def apply_update(params, grads, learning_rate):
    return {
        'w': params['w'] - learning_rate * grads['dw'],
        'b': params['b'] - learning_rate * grads['db'],
    }
