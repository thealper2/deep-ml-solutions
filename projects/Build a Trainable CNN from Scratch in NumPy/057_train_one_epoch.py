def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    losses = []
    current_params = params
    current_opt_state = opt_state
    current_step = step_counter

    for xb, yb in iterate_minibatches(x, y, batch_size, seed):
        current_step += 1
        current_params, current_opt_state, loss = train_step(
            current_params, current_opt_state, xb, yb,
            lr, beta_one, beta_two, eps, current_step
        )
        losses.append(loss)

    return current_params, current_opt_state, current_step, losses
