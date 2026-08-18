import torch
import torch.nn.functional as F

def causal_depthwise_conv1d(x, weight, bias=None):
    """Run a causal depthwise 1-D convolution over a (B, L, E) sequence.

    Args:
        x: (B, L, E) input sequence.
        weight: (E, K) per-channel kernel.
        bias: optional (E,) added after the convolution.

    Returns:
        (B, L, E) output sequence.
    """
    B, L, E = x.shape
    K = weight.shape[1]
    x_perm = x.permute(0, 2, 1)
    padding = (K - 1, 0)
    x_padded = F.pad(x_perm, padding, mode='constant', value=0)
    weight_dw = weight.unsqueeze(1)
    out = F.conv1d(x_padded, weight_dw, bias=bias, groups=E)
    out = out.permute(0 ,2, 1)
    return out