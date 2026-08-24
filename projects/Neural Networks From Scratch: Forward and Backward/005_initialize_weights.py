def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    if scheme == 'he':
      std = np.sqrt(2.0 / in_dim)
      W = np.random.randn(in_dim, out_dim) * std
      b = np.zeros(out_dim)
      return W, b

    elif scheme == 'xavier':
      std = np.sqrt(1.0 / in_dim)
      W = np.random.randn(in_dim, out_dim) * std
      b = np.zeros(out_dim)
      return W, b

    elif scheme == 'xavier_uniform':
      a = np.sqrt(6.0 / (in_dim + out_dim))
      W = np.random.uniform(-a, a, size=(in_dim, out_dim))
      b = np.zeros(out_dim)
      return W, b

    else:
      raise ValueError(f"Unsupported initialization scheme: {scheme}")