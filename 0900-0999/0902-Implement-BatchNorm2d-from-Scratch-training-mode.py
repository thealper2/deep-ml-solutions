import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    gamma = gamma.view(1, -1, 1, 1)
    beta = beta.view(1, -1, 1, 1)
    return gamma * x_hat + beta
