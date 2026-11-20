import torch
import torch.nn.functional as F


def sft_loss(logits, ids, completion_mask):
    """Mean next-token negative log-likelihood over the completion tokens only."""
    shift_logits = logits[:, :-1, :]
    shift_targets = ids[:, 1:]
    shift_mask = completion_mask[:, 1:].to(torch.float32)

    B, T, V = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.reshape(B * T, V),
        shift_targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)

    return (ce * shift_mask).sum() / shift_mask.sum()
