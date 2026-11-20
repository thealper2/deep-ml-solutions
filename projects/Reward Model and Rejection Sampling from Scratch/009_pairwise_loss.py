import torch
import torch.nn.functional as F


def pairwise_loss(r_chosen, r_rejected):
    """Mean of -log sigmoid(r_chosen - r_rejected)."""
    return F.softplus(-(r_chosen - r_rejected)).mean()


def pairwise_accuracy(r_chosen, r_rejected):
    """Fraction of pairs with r_chosen > r_rejected, as a Python float."""
    return float((r_chosen > r_rejected).float().mean().item())
