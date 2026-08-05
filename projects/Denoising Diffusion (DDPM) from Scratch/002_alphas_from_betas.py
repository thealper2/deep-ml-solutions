def alphas_from_betas(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    return alphas
