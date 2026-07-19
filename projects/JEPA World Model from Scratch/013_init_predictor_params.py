def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    action_embed_w = torch.randn(action_dim, latent_dim) * 0.02
    action_embed_w.requires_grad_(True)

    fc1_w = torch.randn(hidden_dim, 2 * latent_dim) * 0.02
    fc1_w.requires_grad_(True)
    fc1_b = torch.zeros(hidden_dim)
    fc1_b.requires_grad_(True)

    fc2_w = torch.randn(latent_dim, hidden_dim) * 0.02
    fc2_w.requires_grad_(True)
    fc2_b = torch.zeros(latent_dim)
    fc2_b.requires_grad_(True)

    return {
        'action_embed_w': action_embed_w,
        'fc1_w': fc1_w,
        'fc1_b': fc1_b,
        'fc2_w': fc2_w,
        'fc2_b': fc2_b,
    }
