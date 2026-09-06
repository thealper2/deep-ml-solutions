def train_model(dataset: dict, cfg: dict, seed: int = 0) -> tuple:
    G = dataset['G']
    T = dataset['tokens'].shape[1]
    vocab_size = 4 + G * G + 1

    params = init_gpt_params(vocab_size, cfg['d_model'], cfg['n_layers'], T, seed)
    dyn = init_dynamics_params(cfg['d_model'], cfg['hidden'], seed)

    all_params = list(params.values()) + list(dyn.values())
    opt = torch.optim.Adam(all_params, lr=cfg['lr'])

    history = []

    for step in range(cfg['steps']):
        batch = get_batch(dataset, cfg['batch_size'], step)
        losses = train_step(
            batch, params, dyn, opt,
            n_heads=cfg['n_heads'],
            d_steps=cfg['d_steps'],
            lam_h=cfg['lam_h'],
            lam_kl=cfg['lam_kl'],
            beta=cfg['beta'],
        )
        history.append(losses)

    return params, dyn, history