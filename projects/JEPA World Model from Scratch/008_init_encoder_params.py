def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    
    conv1_w = torch.randn(16, obs_channels, 3, 3) * 0.1
    conv1_b = torch.zeros(16)

    conv2_w = torch.randn(32, 16, 3, 3) * 0.1
    conv2_b = torch.zeros(32)

    h = ((room_size + 2 - 3) // 2) + 1
    w = ((room_size + 2 - 3) // 2) + 1

    fc_in = 32 * h * w

    fc_w = torch.randn(latent_dim, fc_in) * 0.1
    fc_b = torch.zeros(latent_dim)

    conv1_w.requires_grad_(True)
    conv1_b.requires_grad_(True)
    conv2_w.requires_grad_(True)
    conv2_b.requires_grad_(True)
    fc_w.requires_grad_(True)
    fc_b.requires_grad_(True)

    return {
        'conv1_w': conv1_w,
        'conv1_b': conv1_b,
        'conv2_w': conv2_w,
        'conv2_b': conv2_b,
        'fc_w': fc_w,
        'fc_b': fc_b,
    }
