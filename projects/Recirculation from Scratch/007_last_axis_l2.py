def last_axis_l2(x):
    """Return last-axis L2 norms of x with a kept singleton dimension."""
    return torch.norm(x, dim=-1, keepdim=True)