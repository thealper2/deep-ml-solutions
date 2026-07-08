def evaluate_loss(params, x, y):
    logits, _ = model_forward(params, x)
    return float(cross_entropy_loss(logits, y))
