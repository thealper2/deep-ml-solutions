import torch
import math

def effective_rank(H, tol: float = 1e-12) -> float:
    sv = torch.linalg.svdvals(H)
    sv = sv[sv > tol]

    if len(sv) == 0:
        return 0.0

    sv_norm = sv / sv.sum()

    entropy = -torch.sum(sv_norm * torch.log(sv_norm))

    return float(torch.exp(entropy))