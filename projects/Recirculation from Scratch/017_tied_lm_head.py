def tied_lm_head(h, embedding_weight):
    """Project a residual stream to vocabulary logits with a tied embedding table."""
    return h @ embedding_weight.T