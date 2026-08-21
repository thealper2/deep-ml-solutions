def perplexity(loss):
    """Return exp(loss) for a scalar or tensor NTP cross-entropy."""
    return torch.exp(loss) if isinstance(loss, torch.Tensor) else math.exp(loss)