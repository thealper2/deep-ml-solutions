def prefix_tuning_tradeoff(d_model: int, max_sequence_length: int, l_p: int, l_i: int) -> dict:
    """Compute prefix-tuning parameter count and sequence-length trade-off.

    Args:
        d_model: hidden size of the transformer.
        max_sequence_length: model's maximum context length.
        l_p: number of prefix tokens (prepended).
        l_i: number of infix tokens (inserted).

    Returns:
        dict with keys 'trainable_params', 'effective_seq_len', 'prefix_fraction'.
    """
    total_reserved = l_p + l_i
    trainable_params = d_model * total_reserved

    if total_reserved >= max_sequence_length:
        effective_seq_len = 0
        prefix_fraction = 1.0
    else:
        effective_seq_len = max_sequence_length - total_reserved
        prefix_fraction = total_reserved / max_sequence_length

    return {
        'trainable_params': trainable_params,
        'effective_seq_len': effective_seq_len,
        'prefix_fraction': prefix_fraction,
    }