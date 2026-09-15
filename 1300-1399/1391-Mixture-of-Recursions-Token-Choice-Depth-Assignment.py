import torch


def token_choice_recursion(h, router_w, block, n_recursions):
    h = torch.as_tensor(h, dtype=torch.float32)
    router_w = torch.as_tensor(router_w, dtype=torch.float32)
    T, d = h.shape

    logits = h @ router_w
    probs = torch.softmax(logits, dim=-1)
    depths = torch.argmax(logits, dim=-1) + 1

    gates = probs.gather(1, (depths - 1).unsqueeze(1)).squeeze(1)

    h = h.clone()
    for r in range(1, n_recursions + 1):
        mask = depths >= r
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        f_out = block(h[idx])
        h[idx] = h[idx] + gates[idx].unsqueeze(-1) * f_out

    load = torch.bincount(depths - 1, minlength=n_recursions).to(torch.int64)
    return h, depths.to(torch.int64), load