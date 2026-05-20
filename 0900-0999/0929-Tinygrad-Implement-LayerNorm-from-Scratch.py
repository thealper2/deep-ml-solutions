from tinygrad import Tensor

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
    x_norm = (x - mean) / (var + eps).sqrt()
    return gamma * x_norm + beta