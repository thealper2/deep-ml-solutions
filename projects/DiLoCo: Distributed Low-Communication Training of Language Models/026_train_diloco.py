def train_diloco(init_params, worker_shards, num_rounds, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed=0):
    global_params = clone_params(init_params)
    outer_state = init_outer_optimizer(global_params)
    
    round_losses = []
    
    for round_idx in range(num_rounds):
        round_seed = seed + round_idx
        global_params, outer_state, worker_losses = run_diloco_round(
            global_params, outer_state, worker_shards, num_inner_steps, batch_size,
            inner_hparams, outer_lr, momentum_coef, round_seed
        )
        mean_round_loss = np.mean(worker_losses)
        round_losses.append(mean_round_loss)
    
    return global_params, {'round_losses': round_losses}
