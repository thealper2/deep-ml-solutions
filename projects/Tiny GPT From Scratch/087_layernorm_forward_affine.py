def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_norm = layernorm_forward_normalize(x, mean, var, eps)
    y = gamma * x_norm + beta
    return {
        'y': y, 
        'cache': {
            'x': x, 
            'x_hat': x_norm, 
            'mean': mean, 
            'var': var, 
            'gamma': gamma, 
            'eps': eps
        }
    }
