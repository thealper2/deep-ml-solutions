def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    n = dataset['observations'].shape[0]
    history = []

    for step in range(num_steps):
        indices = torch.randint(0, n, (batch_size,), generator=rng)
        batch = {
            'observations': dataset['observations'][indices],
            'actions': dataset['actions'][indices],
            'next_observations': dataset['next_observations'][indices],
        }

        encoder_params, target_params, predictor_params, loss, col = jepa_training_step(
            batch, encoder_params, target_params, predictor_params, lr, tau
        )
        history.append({'loss': loss, 'collapse': col})

    return encoder_params, target_params, predictor_params, history
