def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)
