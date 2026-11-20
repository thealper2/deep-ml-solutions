import torch
import torch.nn.functional as F


def sft_loss(logits, ids, mask):
    """Mean next-token cross-entropy over the targets whose mask is 1."""
    shift_logits = logits[:, :-1, :]
    shift_targets = ids[:, 1:]
    shift_mask = mask[:, 1:].to(torch.float32)

    B, T, V = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.reshape(B * T, V),
        shift_targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)

    masked = ce * shift_mask
    return masked.sum() / shift_mask.sum()