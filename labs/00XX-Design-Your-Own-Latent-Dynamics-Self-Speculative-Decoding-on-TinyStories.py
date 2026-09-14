import torch
import torch.nn.functional as F
import numpy as np


def init_aux_params(d_model: int, vocab_size: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    hidden = 4 * d_model
    W1 = torch.randn(2 * d_model, hidden) * (1.0 / (2 * d_model) ** 0.5)
    b1 = torch.zeros(hidden)
    W2 = torch.randn(hidden, d_model) * 0.01
    b2 = torch.zeros(d_model)
    return {
        "W1": W1.float().requires_grad_(True),
        "b1": b1.float().requires_grad_(True),
        "W2": W2.float().requires_grad_(True),
        "b2": b2.float().requires_grad_(True),
    }


def latent_transition(h, x_emb, aux):
    inp = torch.cat([h, x_emb], dim=-1)
    z = F.gelu(inp @ aux["W1"] + aux["b1"])
    delta = z @ aux["W2"] + aux["b2"]
    return h + delta


def latent_objective(h, x, mask, params, aux):
    B, T, d = h.shape
    if T < 2:
        return torch.zeros((), dtype=h.dtype, device=h.device)

    wte = params["wte"]
    x_emb_all = wte[x].detach()
    h_det = h.detach()

    loss = torch.zeros((), dtype=h.dtype, device=h.device)
    weights = [1.0, 0.7, 0.4, 0.2]
    n_steps = 4

    cur_h = h_det
    for step in range(n_steps):
        n_pos = T - 1 - step
        if n_pos <= 0:
            break

        h_cur = cur_h[:, :n_pos, :]
        xe_cur = x_emb_all[:, 1 + step:, :]
        h_tgt = h_det[:, 1 + step:, :]

        m = (mask[:, :n_pos] & mask[:, 1 + step:]).float()
        denom = m.sum().clamp_min(1.0)

        if step == 0:
            h_in = h_cur + 0.02 * torch.randn_like(h_cur)
        else:
            h_in = h_cur

        h_pred = latent_transition(h_in, xe_cur, aux)

        diff = ((h_pred - h_tgt) ** 2).mean(dim=-1)
        loss = loss + weights[step] * (diff * m).sum() / denom

        cur_h = h_pred

    return loss
