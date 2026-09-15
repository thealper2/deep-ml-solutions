import torch


def expert_choice_recursion(h, router_w, block, n_recursions, capacity):
    h = torch.as_tensor(h, dtype=torch.float32)
    router_w = torch.as_tensor(router_w, dtype=torch.float32)
    T, d = h.shape

    if isinstance(capacity, int):
        caps = [capacity] * n_recursions
    else:
        caps = list(capacity)

    active = torch.ones(T, dtype=torch.bool)
    depths = torch.zeros(T, dtype=torch.int64)

    for r in range(n_recursions):
        if not active.any():
            break
        
        k = min(caps[r], int(active.sum().item()))
        if k <= 0:
            break

        scores = torch.sigmoid(h @ router_w[r])
        masked = scores.masked_fill(~active, float('-inf'))

        topk_vals, topk_idx = torch.topk(masked, k)

        selected_h = h[topk_idx]
        f_out = block(selected_h)
        gates = torch.sigmoid(h[topk_idx] @ router_w[r]).unsqueeze(-1)

        h = h.clone()
        h[topk_idx] = h[topk_idx] + gates * f_out

        depths[topk_idx] += 1

        new_active = torch.zeros(T, dtype=torch.bool)
        new_active[topk_idx] = True
        active = new_active

    return h, depths