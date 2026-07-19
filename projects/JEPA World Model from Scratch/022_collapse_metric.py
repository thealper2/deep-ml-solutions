def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    stds = torch.std(embeddings, dim=0)
    return torch.mean(stds)
