import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x = np.array(x)
    analytical_grad = np.array(analytical_grad)
    n = len(x)
    numerical_grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += epsilon

        x_minus = x.copy()
        x_minus[i] -= epsilon

        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    norm_num = np.linalg.norm(numerical_grad)
    norm_analytical = np.linalg.norm(analytical_grad)

    if norm_num == 0 and norm_analytical == 0:
        relative_error = 0.0
    else:
        relative_error = np.linalg.norm(numerical_grad - analytical_grad) / (norm_num + norm_analytical)

    return numerical_grad, relative_error