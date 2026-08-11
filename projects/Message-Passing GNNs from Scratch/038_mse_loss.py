def mse_loss(predictions, targets):
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    return ((pred_flat - target_flat) ** 2).mean()
