def sgd_training_step(token_ids, params, lr):
    """Run one vanilla SGD step of next-token prediction and return the loss.

    Args:
        token_ids: (B, L) integer tensor of token ids with L >= 2.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            norm_weight (D,), and blocks (list of nested param dicts).
            Parameter tensors must have requires_grad=True and are updated in place.
        lr: vanilla SGD learning rate.

    Returns:
        Python float, the next-token cross-entropy from this step.
    """
    def zero_grads(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                zero_grads(v)
        elif isinstance(obj, list):
            for v in obj:
                zero_grads(v)
        elif isinstance(obj, torch.Tensor) and obj.requires_grad:
            if obj.grad is not None:
                obj.grad.zero_()

    zero_grads(params)

    logits = mamba_lm_forward(token_ids, params)

    loss = next_token_cross_entropy(logits, token_ids)

    loss.backward()

    def sgd_update(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                sgd_update(v)
        elif isinstance(obj, list):
            for v in obj:
                sgd_update(v)
        elif isinstance(obj, torch.Tensor) and obj.requires_grad and obj.grad is not None:
            obj.data -= lr * obj.grad.data

    sgd_update(params)

    return float(loss.item())
