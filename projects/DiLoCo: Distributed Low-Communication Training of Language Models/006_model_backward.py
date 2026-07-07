def model_backward(params, cache, labels):
    x, z1, h1, logits = cache['x'], cache['z1'], cache['h1'], cache['logits']
    N = x.shape[0]
    C = logits.shape[0]

    probs = softmax(logits)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    dlogits = (probs - one_hot) / N

    db2 = np.sum(dlogits, axis=0)
    dW2 = h1.T @ dlogits
    dh1 = dlogits @ params['W2'].T

    dz1 = dh1 * (z1 > 0)

    db1 = np.sum(dz1, axis=0)
    dW1 = x.T @ dz1

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
