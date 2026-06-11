def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
