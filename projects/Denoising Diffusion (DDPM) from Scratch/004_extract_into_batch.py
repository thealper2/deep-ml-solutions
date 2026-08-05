def extract_into_batch(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return a.gather(0, t.long()).reshape(-1, 1, 1, 1)
