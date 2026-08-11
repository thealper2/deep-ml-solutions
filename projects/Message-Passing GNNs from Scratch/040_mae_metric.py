def mae_metric(predictions, targets):
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    return (pred_flat - target_flat).abs().mean().item()
