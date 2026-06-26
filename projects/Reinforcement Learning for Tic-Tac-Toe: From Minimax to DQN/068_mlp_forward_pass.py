def mlp_forward_pass(params, x):
    """Forward pass through a two-layer MLP with ReLU hidden activation.

    Args:
        params: dict with keys 'W1', 'b1', 'W2', 'b2'.
        x: np.ndarray of shape (batch, input_dim).

    Returns:
        (q_values, cache) where q_values has shape (batch, output_dim) and
        cache is a dict with keys {'x', 'z1', 'h1', 'q'}.
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    z1 = x @ W1 + b1
    h1 = np.maximum(0, z1)
    q = h1 @ W2 + b2

    cache = {'x': x, 'z1': z1, 'h1': h1, 'q': q}
    return q, cache
