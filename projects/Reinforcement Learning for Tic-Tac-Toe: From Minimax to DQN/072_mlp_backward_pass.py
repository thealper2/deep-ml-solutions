def mlp_backward_pass(params, cache, action_indices, target_q):
    """Backprop MSE-on-chosen-action loss through the MLP and return param gradients."""
    x, z1, h1, q = cache['x'], cache['z1'], cache['h1'], cache['q']
    W2 = params['W2']
    batch_size = x.shape[0]

    dq = np.zeros_like(q)
    dq[np.arange(batch_size), action_indices] = 2 * (q[np.arange(batch_size), action_indices] - target_q) / batch_size

    db2 = np.sum(dq, axis=0)
    dW2 = h1.T @ dq

    dh1 = dq @ W2.T
    dz1 = dh1 * (z1 > 0)

    db1 = np.sum(dz1, axis=0)
    dW1 = x.T @ dz1

    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2,
    }
