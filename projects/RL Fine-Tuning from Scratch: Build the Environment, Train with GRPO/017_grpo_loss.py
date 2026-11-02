import torch


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.02):
    A = advantages.unsqueeze(1)

    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    pg = -torch.min(unclipped, clipped)

    kl = kl_k3(logp, ref_logp)

    per_token = pg + beta * kl
    denom = mask.sum()
    loss = (mask * per_token).sum() / denom

    with torch.no_grad():
        pg_mean = float((mask * pg).sum() / denom)
        kl_mean = float((mask * kl).sum() / denom)
        outside = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float()
        clip_frac = float((mask * outside).sum() / denom)

    stats = {"pg": pg_mean, "kl": kl_mean, "clip_frac": clip_frac}
    return stats