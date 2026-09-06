def train_step(batch: dict, params: dict, dyn: dict, opt, n_heads: int, d_steps: int,
               lam_h: float, lam_kl: float, beta: float = 1.0) -> dict:
    opt.zero_grad()

    losses = nextlat_loss(batch, params, dyn, n_heads, d_steps, lam_h, lam_kl, beta)

    losses['total'].backward()

    opt.step()

    return {
        'total': float(losses['total'].item()),
        'next_token': float(losses['next_token'].item()),
        'next_h': float(losses['next_h'].item()),
        'kl': float(losses['kl'].item())
    }
