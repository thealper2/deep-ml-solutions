def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    return betas
