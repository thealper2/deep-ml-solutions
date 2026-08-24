def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    params = {}

    if kind == 'relu':
      def forward(x):
        y = np.maximum(x, 0)
        cache = x
        return y, cache

      def backward(dout, cache):
        x = cache
        dx = dout * (x > 0)
        param_grads = {}
        return dx, param_grads

      return {
        'params': params,
        'forward': forward,
        'backward': backward
      }

    elif kind == "tanh":
      def forward(x):
        y = np.tanh(x)
        cache = y
        return y, cache
      
      def backward(dout, cache):
        y = cache
        dx = dout * (1 - y * y)
        param_grads = {}
        return dx, param_grads

      return {
        'params': params,
        'forward': forward,
        'backward': backward
      }

    elif kind == 'sigmoid':
      def forward(x):
        y = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        cache = y
        return y, cache

      def backward(dout, cache):
        y = cache
        dx = dout * y * (1 - y)
        param_grads = {}
        return dx, param_grads

      return {
        'params': params,
        'forward': forward,
        'backward': backward
      }

    else:
      raise ValueError(f"Unsupported activation: {kind}")