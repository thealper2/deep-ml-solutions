import math
import torch
import torch.nn as nn


def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float)
    theta = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin

def apply_rope(x, cos, sin):
    head_dim = x.shape[-1]
    d = head_dim // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos_expanded = torch.cat([cos, cos], dim=-1)
    sin_expanded = torch.cat([sin, sin], dim=-1)
    return x * cos_expanded + rotated * sin_expanded