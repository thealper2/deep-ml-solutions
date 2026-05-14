import torch

def grad_of_quadratic(x_value: float) -> float:
    t = torch.tensor(x_value, requires_grad=True)
    y = t ** 2 + 3 * t + 2
    y.backward()
    return float(t.grad)