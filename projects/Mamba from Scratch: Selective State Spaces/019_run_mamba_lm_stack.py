def run_mamba_lm_stack(embeddings, params):
    """Run token embeddings through stacked Mamba residual blocks and a final RMSNorm.

    Args:
        embeddings: (B, L, D) token embeddings.
        params: dict with key `blocks` (list of per-block dicts for `mamba_block`)
            and key `norm_weight` of shape (D,) for the final RMSNorm (eps=1e-5).

    Returns:
        (B, L, D) hidden states after the stack and final RMSNorm.
    """
    x = embeddings
    blocks = params["blocks"]

    for block_params in blocks:
        x = mamba_block(x, block_params)

    norm_weight = params["norm_weight"]
    x = rms_norm(x, norm_weight)
    return x