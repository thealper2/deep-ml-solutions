import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    N, C, H, W = X.shape
    G = num_groups
    x_reshaped = X.reshape(N, G, C // G, H, W)
    mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    x_norm = (x_reshaped - mean) / np.sqrt(var + epsilon)
    x_norm = x_norm.reshape(N, C, H, W)
    return x_norm * gamma + beta