def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    matrix = make_2d_random(block_size, d_model, seed=None)
    scaled_matrix = scale_w_small(matrix, scale)
    return scaled_matrix
