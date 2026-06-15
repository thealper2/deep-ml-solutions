def mse_loss_and_grad(y_pred, y_true):
    n_elements = y_pred.size
    error = y_pred - y_true
    loss = np.mean(error ** 2)
    dy_pred = 2 * error / n_elements
    return loss, dy_pred
