def softmax_cross_entropy_forward(logits, y, eps=1e-12):
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, y, eps)
    return loss if loss != -0.0 else 0.0
