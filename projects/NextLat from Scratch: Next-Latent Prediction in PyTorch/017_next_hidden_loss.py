import torch
import torch.nn.functional as F

def next_hidden_loss(h, h_hats: list, mask, beta: float = 1.0):
    if not h_hats:
        return torch.tensor(0.0, device=h.device)

    B, T, d = h.shape
    d_steps = len(h_hats)
    losses = []

    for step_idx, h_hat in enumerate(h_hats):
        i = step_idx + 1
        target = h[:, i:T - d_steps + i].detach()
        mask_slice = mask[:, i:T - d_steps + i]
        loss_elem = F.smooth_l1_loss(h_hat, target, reduction='none', beta=beta)
        loss_feat_avg = loss_elem.mean(dim=-1)
        masked_loss = loss_feat_avg[mask_slice]
        if masked_loss.numel() > 0:
            losses.append(masked_loss.mean())

    if not losses:
        return torch.tensor(0.0, device=h.device)

    return torch.stack(losses).mean()