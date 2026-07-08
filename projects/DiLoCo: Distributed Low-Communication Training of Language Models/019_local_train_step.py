def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    logits, cache = model_forward(params, x_batch)
    loss = cross_entropy_loss(logits, y_batch)
    grads = model_backward(params, cache, y_batch)
    adam_state = update_adam_moments(adam_state, grads, beta1, beta2)
    m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)
    new_params = adam_param_step(params, m_hat, v_hat, lr, eps)
    new_params = decoupled_weight_decay(new_params, lr, weight_decay)
    return new_params, adam_state, loss
