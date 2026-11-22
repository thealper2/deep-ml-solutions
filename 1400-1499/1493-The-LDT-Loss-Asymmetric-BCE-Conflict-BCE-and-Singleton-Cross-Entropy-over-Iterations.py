import torch
import torch.nn.functional as F


def asymmetric_bce(logits, target, w_pos, w_neg):
    target = target.to(logits.dtype)
    pos = w_pos * target * F.logsigmoid(logits)
    neg = w_neg * (1 - target) * F.logsigmoid(-logits)
    return -(pos + neg).mean()


def conflict_bce(cls_logits, is_bottom):
    return F.binary_cross_entropy_with_logits(cls_logits, is_bottom.to(cls_logits.dtype))


def singleton_ce(logits, target):
    counts = target.sum(dim=-1)
    mask = counts == 1
    if mask.sum() == 0:
        return logits.sum() * 0.0

    B, n, _, V = logits.shape
    sel_logits = logits[mask]
    sel_target = target[mask].to(torch.long).argmax(dim=-1)
    return F.cross_entropy(sel_logits, sel_target)


def target_is_bottom(y_hat):
    return (~y_hat.any(dim=-1)).flatten(1).any(dim=-1)


def ldt_loss(b_iters, c_iters, y_hat, w_pos, w_neg, lam_cls, lam_ce):
    L = b_iters.shape[0]
    is_bottom = target_is_bottom(y_hat)

    total = 0.0
    for i in range(L):
        b = b_iters[i]
        c = c_iters[i]
        total = total + (
            asymmetric_bce(b, y_hat, w_pos, w_neg)
            + lam_cls * conflict_bce(c, is_bottom)
            + lam_ce * singleton_ce(b, y_hat)
        )

    return total / L
