import torch
import torch.nn.functional as F

def ntp_loss(logits, tokens):
    logits_shifted = logits[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    logits_flat = logits_shifted.reshape(-1, logits.shape[-1])
    tokens_flat = tokens_shifted.reshape(-1)
    loss = F.cross_entropy(logits_flat, tokens_flat, reduction='mean')
    return loss