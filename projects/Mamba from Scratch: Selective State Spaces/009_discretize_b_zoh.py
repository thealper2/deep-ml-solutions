def discretize_b_zoh(delta, a, b):
    """Discretize B with the exact diagonal zero-order-hold formula.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        b_bar: (batch, seq_len, d_inner, d_state) discrete B.
    """
    delta_exp = delta.unsqueeze(-1)
    exp_delta_a = torch.exp(delta_exp * a)
    factor = (exp_delta_a - 1) / a
    b_exp = b.unsqueeze(-2)
    b_bar = factor * b_exp
    return b_bar