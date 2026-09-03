import torch
import torch.nn as nn


def all_reduce_grads(replicas):
    all_params = [list(m.parameters()) for m in replicas]

    for params in zip(*all_params):
        grads = [p.grad for p in params]
        valid_grads = [g for g in grads if g is not None]

        if not valid_grads:
            continue

        avg_grad = torch.stack(valid_grads).mean(dim=0)

        for p in params:
            if p.grad is not None:
                p.grad.copy_(avg_grad)
            else:
                p.grad = avg_grad.clone()