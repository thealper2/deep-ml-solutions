def next_token_cross_entropy(logits, token_ids):
    """Compute the mean next-token cross-entropy from logits and token ids."""
    B, T, V = logits.shape
    logits_shifted = logits[:, :-1, :]
    targets_shifted = token_ids[:, 1:]
    logits_flat = logits_shifted.reshape(-1, V)
    targets_flat = targets_shifted.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss