from tinygrad import Tensor

def clip_grad_norm(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return 0.0

    squared_norms = [(g.detach() * g.detach()).sum() for g in grads]
    total_norm = Tensor.stack(squared_norms).sum().sqrt()

    total_norm_value = total_norm.item()

    if total_norm_value > max_norm:
        scale = max_norm / (total_norm_value + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.assign(p.grad * scale)

    return total_norm_value