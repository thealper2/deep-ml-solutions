import numpy as np

class Sequential:
    """Ordered container of layers with a flat parameters() view."""
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
