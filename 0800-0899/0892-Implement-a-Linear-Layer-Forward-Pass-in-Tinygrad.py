from tinygrad import Tensor

def linear_forward(x: Tensor, W: Tensor, b: Tensor) -> Tensor:
    return x @ W.T + b