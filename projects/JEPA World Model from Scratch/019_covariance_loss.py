def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    B, D = embeddings.shape
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (B - 1)
    off_diag = cov * (1 - torch.eye(D, device=cov.device))
    loss = (off_diag ** 2).sum() / D
    return loss
