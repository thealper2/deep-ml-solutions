import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return 0.0

    total_norm = torch.norm(
        torch.stack([g.detach().norm(2) for g in grads]), 2
    )

    total_norm_value = total_norm.item()

    if total_norm_value > max_norm:
        scale = max_norm / (total_norm_value + 1e-6)
        for g in grads:
            g.mul_(scale)

    return total_norm_value