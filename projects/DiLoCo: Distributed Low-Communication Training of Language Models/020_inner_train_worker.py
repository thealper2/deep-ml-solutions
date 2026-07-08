def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    worker_params = clone_params(params)
    adam_state = init_adamw_state(worker_params)
    rng = np.random.default_rng(seed)
    total_loss = 0.0

    for _ in range(num_inner_steps):
        x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
        worker_params, adam_state, loss = local_train_step(worker_params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay)
        total_loss += loss

    mean_loss = total_loss / num_inner_steps
    return worker_params, mean_loss
