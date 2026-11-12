import torch

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0, generator=None):
        """Kaiming-initialised weight (fan_in, fan_out) and an optional zero bias."""
        self.weight = torch.randn((fan_in, fan_out), generator=generator) * (gain / fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        """x @ weight (+ bias), also stored in self.out."""
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] if self.bias is None else [self.weight, self.bias]
