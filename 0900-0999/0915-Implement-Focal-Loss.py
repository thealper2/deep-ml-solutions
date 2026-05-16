import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pt = probs[torch.arange(len(targets)), targets]
    log_pt = log_probs[torch.arange(len(targets)), targets]
    focal_weight = (1 - pt) ** gamma
    loss = -focal_weight * log_pt
    return loss.mean()