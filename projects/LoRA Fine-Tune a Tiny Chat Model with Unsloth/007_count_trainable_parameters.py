def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params
