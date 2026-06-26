def dqn_train_step(online_params, target_params, adam_state, buffer, batch_size, gamma, lr, rng):
    """Run one DQN minibatch update. Return (online_params, adam_state, loss)."""
    batch = sample_minibatch_from_buffer(buffer, batch_size, rng)
    targets = compute_target_q_with_target_network(target_params, batch, gamma)
    q_values, cache = mlp_forward_pass(online_params, batch['states'])
    loss = mse_loss_on_chosen_action(q_values, batch['actions'], targets)
    grads = mlp_backward_pass(online_params, cache, batch['actions'], targets)
    new_online_params, new_adam_state = adam_update_step(online_params, grads, adam_state, lr)
    return new_online_params, new_adam_state, loss
