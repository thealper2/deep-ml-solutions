def local_shard_adam_update(params, grads, worker_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    t = worker_state['t'] + 1
    worker_state['t'] = t

    m = worker_state['m']
    v = worker_state['v']
    shard_slices = worker_state['shard_slices']
    shapes = worker_state['shapes']

    updated_shards = {}

    for key in params:
        start, end = shard_slices[key]
        param_shard = params[key].flatten()[start:end]
        grad_shard = grads[key].flatten()[start:end]
        m_new = beta1 * m[key] + (1 - beta1) * grad_shard
        m[key] = m_new
        v_new = beta2 * v[key] + (1 - beta2) * (grad_shard ** 2)
        v[key] = v_new
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        updated_shard = param_shard - lr * m_hat / (np.sqrt(v_hat) + eps)
        updated_shards[key] = updated_shard

    return updated_shards, worker_state 
