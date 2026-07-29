import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    max_val = t.max(dim=dim, keepdim=True)[0]
    exp_t = torch.exp(t - max_val)
    sum_exp = exp_t.sum(dim=dim, keepdim=True)
    return exp_t / sum_exp
