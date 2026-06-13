def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    return x_heads_back.shape[1]
