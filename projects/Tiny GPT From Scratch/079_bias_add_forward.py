def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    y = vector_matrix_broadcast_add(x, b)
    return {'y': y, 'cache': {'b_shape': b.shape}}
