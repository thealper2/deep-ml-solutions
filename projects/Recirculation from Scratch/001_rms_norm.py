def rms_norm(x, gain, eps=1e-6):
    """Apply RMSNorm over the last dimension with a learnable gain vector and eps 1e-6."""
    variance = x.pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + eps)
    return (x * rsqrt) * gain
