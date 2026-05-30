from tinygrad import Tensor

def batchnorm2d(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdim=True)
    x_hat = (x - mean) / (var + eps).sqrt()
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)
    return gamma * x_hat + beta