def make_optimizer(model, lr, param, base_width=BASE_WIDTH):
    """Adam with the input layer in one group and the other weights, scaled for muP, in another."""
    width = model[0].out_features
    if param == "mup":
        hidden_lr = lr * base_width / width
    else:
        hidden_lr = lr

    return torch.optim.Adam([
        {"params": [model[0].weight], "lr": lr},
        {"params": [m.weight for m in model[1:] if hasattr(m, "weight")], "lr": hidden_lr}
    ])