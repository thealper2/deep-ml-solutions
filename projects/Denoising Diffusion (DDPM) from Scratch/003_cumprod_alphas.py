def cumprod_alphas(alphas: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod
