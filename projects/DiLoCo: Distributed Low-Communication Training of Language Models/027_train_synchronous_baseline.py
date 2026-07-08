def train_synchronous_baseline(init_params, worker_shards, num_steps, batch_size, inner_hparams, seed=0):
    params = clone_params(init_params)
    
    adam_state = init_adamw_state(params)
    
    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']
    
    rng = np.random.default_rng(seed)
    
    step_losses = []
    num_workers = len(worker_shards)
    
    for step in range(num_steps):
        grad_list = []
        loss_list = []
        
        for worker_idx, (x_shard, y_shard) in enumerate(worker_shards):
            x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
            logits, cache = model_forward(params, x_batch)
            loss = cross_entropy_loss(logits, y_batch)
            loss_list.append(loss)
            grads = model_backward(params, cache, y_batch)
            grad_list.append(grads)
        
        avg_grads = average_params(grad_list)
        adam_state = update_adam_moments(adam_state, avg_grads, beta1, beta2)
        m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)
        params = adam_param_step(params, m_hat, v_hat, lr, eps)
        params = decoupled_weight_decay(params, lr, weight_decay)
        step_losses.append(np.mean(loss_list))
    
    return params, {'step_losses': step_losses}
