def compare_constant_vs_selective_delta(x, a, b, c, delta_const, delta_sel):
    """Compare SSM scan outputs under a constant Delta versus a selective Delta.

    x: (batch, seq_len, d_inner)
    a: (d_inner, d_state) strictly negative continuous diagonal A
    b: (batch, seq_len, d_state)
    c: (batch, seq_len, d_state)
    delta_const: (batch, seq_len, d_inner) non-selective timestep
    delta_sel: (batch, seq_len, d_inner) input-dependent timestep

    Returns:
        y_const: (batch, seq_len, d_inner)
        y_sel: (batch, seq_len, d_inner)
    """
    B, L, E = x.shape
    d_state = a.shape[1]
    a_bar_const = discretize_a_zoh(delta_const, a)
    b_bar_const = discretize_b_zoh(delta_const, a, b)
    a_bar_sel = discretize_a_zoh(delta_sel, a)
    b_bar_sel = discretize_b_zoh(delta_sel, a, b)
    y_const, _ = selective_scan(x, a_bar_const, b_bar_const, c)
    y_sel, _ = selective_scan(x, a_bar_sel, b_bar_sel, c)
    return y_const, y_sel