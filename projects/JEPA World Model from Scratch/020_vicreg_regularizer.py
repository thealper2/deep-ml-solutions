def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    var_loss = variance_loss(embeddings, gamma)
    cov_loss = covariance_loss(embeddings)
    return var_weight * var_loss + cov_weight * cov_loss
