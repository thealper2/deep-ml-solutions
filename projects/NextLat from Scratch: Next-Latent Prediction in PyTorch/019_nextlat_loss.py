def nextlat_loss(batch: dict, params: dict, dyn: dict, n_heads: int, d_steps: int,
                 lam_h: float, lam_kl: float, beta: float = 1.0) -> dict:
    x = batch['x']
    y = batch['y']
    mask = batch['mask']

    h = gpt_hidden_states(x, params, n_heads)
    logits = output_head(h, params)
    next_token = next_token_loss(logits, y, mask)

    eos = params['head_b'].shape[0] - 1
    mask_x = x != eos

    if d_steps > 0:
        h_hats = rollout_latents(h, x, params, dyn, d_steps)
        next_h = next_hidden_loss(h, h_hats, mask_x, beta=beta)
        kl = kl_alignment_loss(h, h_hats, mask_x, params)
    else:
        next_h = torch.tensor(0.0, device=h.device)
        kl = torch.tensor(0.0, device=h.device)

    total = next_token + lam_h * next_h + lam_kl * kl

    return {
        'total': total,
        'next_token': next_token,
        'next_h': next_h,
        'kl': kl,
    }