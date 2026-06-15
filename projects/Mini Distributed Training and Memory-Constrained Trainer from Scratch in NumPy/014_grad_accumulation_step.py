def grad_accumulation_step(x, y, params, micro_batch_size):
    micro_batches = split_into_micro_batches(x, y, micro_batch_size)
    num_micro_batches = len(micro_batches)
    accum_grads = None

    for x_mb, y_mb in micro_batches:
        y_pred, cache = mlp_forward(x_mb, params)
        _, dy_pred = mse_loss_and_grad(y_pred, y_mb)
        grads = mlp_backward(dy_pred, cache, params)
        accum_grads = accumulate_gradients(accum_grads, grads)

    scaled_grads = scale_accumulated_gradients(accum_grads, num_micro_batches)
    return scaled_grads
