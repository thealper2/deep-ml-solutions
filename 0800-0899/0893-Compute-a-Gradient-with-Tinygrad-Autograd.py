from tinygrad import Tensor

def grad_of_quadratic(x_value: float) -> float:
    x = Tensor(x_value, requires_grad=True)
    y = x * x + 3 * x + 2
    y.backward()
    return x.grad.numpy()