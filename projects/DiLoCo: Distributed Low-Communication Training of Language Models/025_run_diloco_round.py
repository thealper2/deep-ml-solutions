def run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']
    
    worker_params_list = []
    worker_losses = []
    
    for worker_idx, (x_shard, y_shard) in enumerate(worker_shards):
        worker_seed = seed + worker_idx
        worker_params, mean_loss = inner_train_worker(
            global_params, x_shard, y_shard, num_inner_steps, batch_size,
            lr, beta1, beta2, eps, weight_decay, worker_seed
        )
        worker_params_list.append(worker_params)
        worker_losses.append(mean_loss)
    
    outer_grad = compute_outer_gradient(global_params, worker_params_list)
    outer_state = update_outer_momentum(outer_state, outer_grad, momentum_coef)
    new_global_params = nesterov_param_update(global_params, outer_state, outer_grad, outer_lr, momentum_coef)
    return new_global_params, outer_state, worker_losses
