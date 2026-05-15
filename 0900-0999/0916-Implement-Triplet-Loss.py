import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()