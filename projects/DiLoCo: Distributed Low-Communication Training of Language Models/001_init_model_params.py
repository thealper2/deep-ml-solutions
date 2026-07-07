def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    np.random.seed(seed=seed)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
