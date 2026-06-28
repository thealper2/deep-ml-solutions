class MLP:
    """Two-layer MLP: Linear -> relu -> Linear."""
    def __init__(self, in_features, hidden, out_features, seed=None):
        self.l1 = Linear(in_features, hidden, seed=seed)
        self.l2 = Linear(hidden, out_features, seed=seed)
        self._relu = bind_unary_tensor_methods()['relu']

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = tensor_from_data(x)

        h = self.l1(x)
        h = self._relu(h)
        return self.l2(h)

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()
