def kl_alignment_loss(h, h_hats: list, mask, params: dict):
    if not h_hats:
        return torch.tensor(0.0, device=h.device)

    B, T, d = h.shape
    d_steps = len(h_hats)
    losses = []

    head_w = params['head_w'].detach()
    head_b = params['head_b'].detach()

    for step_idx, h_hat in enumerate(h_hats):
        i = step_idx + 1
        h_true = h[:, i:T - d_steps +i].detach()
        logits_true = h_true @ head_w + head_b
        logits_pred = h_hat @ head_w + head_b
        log_probs_true = F.log_softmax(logits_true, dim=-1)
        log_probs_pred = F.log_softmax(logits_pred, dim=-1)
        kl_elem = (log_probs_true.exp() * (log_probs_true - log_probs_pred)).sum(dim=-1)
        mask_slice = mask[:, i:T - d_steps + i]
        masked_kl = kl_elem[mask_slice]
        if masked_kl.numel() > 0:
            losses.append(masked_kl.mean())

    if not losses:
        return torch.tensor(0.0, device=h.device)

    return torch.stack(losses).mean()