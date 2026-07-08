def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
