import torch


def act_halting(halt_probs, threshold=0.99):
    halt_probs = torch.as_tensor(halt_probs, dtype=torch.float32)
    T, S = halt_probs.shape

    cumsum = torch.cumsum(halt_probs, dim=1)
    reached = cumsum >= threshold

    any_reached = reached.any(dim=1)
    first_idx = torch.where(
        any_reached,
        reached.float().argmax(dim=1),
        torch.full((T,), S - 1, dtype=torch.long),
    )

    idx_before = (first_idx - 1).clamp_min(0)
    sum_before = torch.gather(cumsum, 1, idx_before.unsqueeze(1)).squeeze(1)
    sum_before = torch.where(first_idx == 0, torch.zeros_like(sum_before), sum_before)

    remainder = 1.0 - sum_before

    idx_range = torch.arange(S).unsqueeze(0).expand(T, S)
    N = first_idx.unsqueeze(1)

    weights = torch.where(idx_range < N, halt_probs, torch.zeros_like(halt_probs))
    weights = torch.where(idx_range == N, remainder.unsqueeze(1), weights)

    n_steps = (first_idx + 1).to(torch.int64)
    ponder = n_steps.float() + remainder
    return weights, n_steps, ponder

def act_output(states, weights):
    states = torch.as_tensor(states, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32)
    return (weights.unsqueeze(-1) * states).sum(dim=1)