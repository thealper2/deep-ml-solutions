def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    var = torch.var(embeddings, dim=0, unbiased=True)
    std = torch.sqrt(var + eps)
    loss = torch.mean(torch.relu(gamma - std))
    return loss
