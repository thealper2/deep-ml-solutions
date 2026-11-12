import torch

class Tanh:
    def __call__(self, x):
        """tanh(x), also stored in self.out."""
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


def activation_stats(t, threshold=0.97):
    """{'mean': float, 'std': float, 'saturated': percent of |t| > threshold}."""
    total = t.numel()
    saturated = float((t.abs() > threshold).sum().item()) / total * 1000.0
    return {
        'mean': float(t.mean().item()),
        'std': float(t.std().item()),
        'saturated': saturated,
    }