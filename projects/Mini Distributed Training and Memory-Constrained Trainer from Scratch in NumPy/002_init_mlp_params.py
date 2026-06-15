def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    np.random.seed(seed=seed)
    W1 = np.random.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(out_dim)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
