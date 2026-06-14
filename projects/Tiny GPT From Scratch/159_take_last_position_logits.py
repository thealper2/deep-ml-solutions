def take_last_position_logits(logits):
    """Return logits at the final time step with shape (1, vocab_size)."""
    return logits[:, -1, :]
