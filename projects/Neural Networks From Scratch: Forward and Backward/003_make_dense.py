def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': W, 'b': b}

    def forward(x):
      y = x @ W + b
      cache = x
      return y, cache

    def backward(dout, cache):
      x = cache
      dx = dout @ W.T
      dW = x.T @ dout
      db = np.sum(dout, axis=0)
      grads = {'W': dW, 'b': db}
      return dx, grads

    return {
      'params': params,
      'forward': forward,
      'backward': backward,
    }