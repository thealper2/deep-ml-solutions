def initialize_mlp_parameters(architecture, seed=0):
    """Initialize MLP weights with He init and zero biases.

    architecture: dict from build_mlp_architecture with input_dim, hidden_dim, output_dim.
    seed: int seed for numpy RNG.
    Returns dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    np.random.seed(seed)
    input_dim = architecture['input_dim']
    hidden_dim = architecture['hidden_dim']
    output_dim = architecture['output_dim']
    
    std1 = np.sqrt(2.0 / input_dim)
    W1 = np.random.randn(input_dim, hidden_dim) * std1

    std2 = np.sqrt(2.0 / hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * std2

    b1 = np.zeros(hidden_dim)
    b2 = np.zeros(output_dim)

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
