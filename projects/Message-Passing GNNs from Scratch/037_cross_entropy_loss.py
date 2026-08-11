def cross_entropy_loss(logits, targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    loss_per_sample = -log_probs[torch.arange(logits.shape[0]), targets]
    return loss_per_sample.mean()
