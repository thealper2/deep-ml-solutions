def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    return d_y.copy(), d_y.copy()
