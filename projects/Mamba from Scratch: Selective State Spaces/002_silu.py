def silu(x):
    """Apply the SiLU activation elementwise."""
    return x * torch.sigmoid(x)