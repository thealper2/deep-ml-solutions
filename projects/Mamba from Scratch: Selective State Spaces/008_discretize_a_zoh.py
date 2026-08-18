def discretize_a_zoh(delta, a):
    """Discretize a diagonal continuous state matrix with zero-order hold.

    delta: torch tensor of shape (..., d)
    a: torch tensor of shape (d, n)
    Returns a_bar of shape (..., d, n).
    """
    delta_expanded = delta.unsqueeze(-1)
    a_bar = torch.exp(delta_expanded * a)
    return a_bar