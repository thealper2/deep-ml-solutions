class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        """Pass x through every layer in order."""
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        """Every layer's parameters, in one flat list."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params