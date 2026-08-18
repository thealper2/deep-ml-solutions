def siso_state_update(h_prev, a_bar, b_bar, c, x_t):
    """Apply one SISO state update and return the scalar readout."""
    h_t = a_bar * h_prev + b_bar * x_t
    y_t = (c * h_t).sum()
    return y_t, h_t