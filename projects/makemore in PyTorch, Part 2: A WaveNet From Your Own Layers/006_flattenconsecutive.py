
import torch

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        """(B, T, C) -> (B, T // n, C * n), dropping a middle dimension of 1."""
        B, T, C = x.shape
        n = self.n
        out = x.reshape(B, T // n, C * n)
        if out.shape[1] == 1:
            out = out.squeeze(1)
        self.out = out
        return self.out

    def parameters(self):
        return []