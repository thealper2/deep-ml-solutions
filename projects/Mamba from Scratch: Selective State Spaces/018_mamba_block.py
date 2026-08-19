def mamba_block(x, params):
    """Apply a pre-norm residual Mamba block to a token sequence.

    Args:
        x: (B, L, D) hidden sequence.
        params: dict with norm_weight (D,) plus every mamba_mixer key.

    Returns:
        (B, L, D) block output.
    """
    norm_weight = params["norm_weight"]
    x_norm = rms_norm(x, norm_weight)
    mixer_out = mamba_mixer(x_norm, params)
    out = x + mixer_out
    return out