def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    w = torch.randn(state_dim, latent_dim) * 0.01
    b = torch.zeros(state_dim)
    return {'w': w, 'b': b}
