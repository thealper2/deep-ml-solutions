import torch

def grad_of_square(x_val):
    """Return dy/dx for y = x**2 at x = x_val using autograd.

    Args:
        x_val (float): scalar input value.

    Returns:
        float: gradient of x**2 w.r.t. x at x_val.
    """
    x = torch.tensor(x_val, requires_grad=True)
    y = x ** 2
    y.backward()
    return x.grad.item()
