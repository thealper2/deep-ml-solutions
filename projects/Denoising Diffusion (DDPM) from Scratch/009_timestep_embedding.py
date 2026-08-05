import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    half = dim // 2
    B = t.shape[0]

    if half == 0:
        return torch.zeros(B, dim, dtype=torch.float32)

    if half == 1:
        freqs = torch.ones(half, dtype=torch.float32)
    else:
        freqs = 10000.0 ** (torch.arange(half, dtype=torch.float32) / (half - 1))

    angles = t.float()[:, None] / freqs[None, :]

    sin_emb = torch.sin(angles)
    cos_emb = torch.cos(angles)

    emb = torch.cat([sin_emb, cos_emb], dim=-1)

    return emb
