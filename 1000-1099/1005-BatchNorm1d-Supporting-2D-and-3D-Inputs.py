import numpy as np

def batchnorm1d(x, gamma, beta, running_mean, running_var, training=True, momentum=0.1, eps=1e-5):
    """
    BatchNorm1d supporting 2D (N, C) and 3D (N, L, C) inputs.

    Returns a dict with keys 'out', 'running_mean', 'running_var'.
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    running_mean = np.array(running_mean)
    running_var = np.array(running_var)

    if x.ndim == 2:
        reduce_axes = (0,)
    elif x.ndim == 3:
        reduce_axes = (0, 1)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    if training:
        mu = np.mean(x, axis=reduce_axes, keepdims=True)
        var = np.var(x, axis=reduce_axes, keepdims=True, ddof=0)
        x_hat = (x - mu) / np.sqrt(var + eps)
        out = gamma.reshape(1, -1) * x_hat + beta.reshape(1, -1)
        running_mean = (1 - momentum) * running_mean + momentum * mu.squeeze()
        running_var = (1 - momentum) * running_var + momentum * var.squeeze()
    else:
        mu = running_mean.reshape(1, -1)
        var = running_var.reshape(1, -1)
        x_hat = (x - mu) / np.sqrt(var + eps)
        out = gamma.reshape(1, -1) * x_hat + beta.reshape(1, -1)
        running_mean = running_mean
        running_var = running_var
    
    return {
        "out": out.tolist(),
        "running_mean": running_mean.tolist(),
        "running_var": running_var.tolist(),
    }