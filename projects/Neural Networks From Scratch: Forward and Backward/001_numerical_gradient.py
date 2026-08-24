import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    if x.size == 0:
        return grad
    
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy()
        x_plus[idx] += eps
        x_minus = x.copy()
        x_minus[idx] -= eps
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)
        it.iternext()

    return grad