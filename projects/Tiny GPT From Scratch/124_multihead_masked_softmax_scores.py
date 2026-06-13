def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    B, n_heads, T, T = scores.shape
    scores_2d = scores.reshape(-1, T)
    mask_expanded = mask.reshape(1, 1, T, T)
    masked_scores = apply_causal_mask(scores, mask_expanded)
    masked_scores_2d = masked_scores.reshape(-1, T)
    probs_2d = stable_softmax_2d_rowwise(masked_scores_2d)
    probs = probs_2d.reshape(B, n_heads, T, T)
    return probs
