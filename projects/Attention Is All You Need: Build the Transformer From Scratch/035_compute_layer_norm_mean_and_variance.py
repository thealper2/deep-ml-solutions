import torch

def compute_layer_norm_mean_and_variance(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return mean, var
