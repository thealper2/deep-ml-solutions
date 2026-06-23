import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, var = compute_layer_norm_mean_and_variance(x)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
