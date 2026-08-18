def compare_euler_zoh_b(delta, a, b):
    """Compare exact ZOH discrete B to the Euler shortcut.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        dict with keys 'b_bar_zoh', 'b_bar_euler', and 'abs_diff', each
        of shape (batch, seq_len, d_inner, d_state).
    """
    b_bar_zoh = discretize_b_zoh(delta, a, b)
    b_bar_euler = delta.unsqueeze(-1) * b.unsqueeze(2)
    abs_diff = torch.abs(b_bar_zoh - b_bar_euler)

    return {
        'b_bar_zoh': b_bar_zoh,
        'b_bar_euler': b_bar_euler,
        'abs_diff': abs_diff,
    }