import torch
import torch.nn.functional as F

def next_token_loss(logits, targets, mask):
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)

    ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    masked_ce = ce[mask_flat]
    if masked_ce.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    return masked_ce.mean()