import torch

def rms_norm(x, weight, eps=1e-5):
    """Normalize a hidden sequence with RMSNorm using a learned per-channel scale."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight