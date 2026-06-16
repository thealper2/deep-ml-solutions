def compute_local_gradients(x, y, params):
    """Compute parameter gradients for one worker's data shard.

    Forward (mlp_forward) -> loss gradient (mse_loss_and_grad) -> backward
    (mlp_backward). Return a grads dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    y_pred, cache = mlp_forward(x, params)
    _, dy_pred = mse_loss_and_grad(y_pred, y)
    grads = mlp_backward(dy_pred, cache, params)
    return grads
