import torch

def bn_eval(x, mean, var, gamma, beta, eps=1e-5):
    """Apply batch-norm inference normalization.

    Args:
        x (Tensor): input tensor
        mean (Tensor): running mean
        var (Tensor): running variance
        gamma (Tensor): scale parameter
        beta (Tensor): shift parameter
        eps (float): numerical stability constant

    Returns:
        Tensor: normalized and affine-transformed tensor
    """
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta
