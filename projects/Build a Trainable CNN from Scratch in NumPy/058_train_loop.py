def train_loop(params, x_train, y_train, num_epochs, batch_size, lr=1e-3, beta_one=0.9, beta_two=0.999, eps=1e-8, seed=0):
    opt_state = {}
    for layer_name in params:
        opt_state[layer_name] = {}
        for param_name in params[layer_name]:
            param = params[layer_name][param_name]
            opt_state[layer_name][param_name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param)
            }
    
    current_params = params
    current_opt_state = opt_state
    step_counter = 0
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_seed = seed + epoch
        current_params, current_opt_state, step_counter, losses = train_one_epoch(
            current_params, current_opt_state, x_train, y_train, batch_size,
            lr, beta_one, beta_two, eps, step_counter, epoch_seed
        )
        loss_history.extend(losses)
    
    return current_params, loss_history
