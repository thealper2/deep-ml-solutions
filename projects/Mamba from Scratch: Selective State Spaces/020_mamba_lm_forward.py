def mamba_lm_forward(token_ids, params):
    """Map token ids through embeddings, the Mamba stack, and an LM head.

    Args:
        token_ids: (B, L) integer tensor of token ids.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            blocks (list), and norm_weight (D,).

    Returns:
        (B, L, V) logits.
    """
    embed_weight = params["embed_weight"]
    embeddings = embed_weight[token_ids]
    hidden = run_mamba_lm_stack(embeddings, params)
    lm_head_weight = params["lm_head_weight"]
    logits = hidden @ lm_head_weight.T
    return logits