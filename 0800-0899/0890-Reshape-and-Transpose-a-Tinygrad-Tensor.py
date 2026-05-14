from tinygrad import Tensor

def flatten_then_reshape(x: Tensor, new_shape) -> Tensor:
    return x.flatten().reshape(new_shape)

def transpose_last_two(x: Tensor) -> Tensor:
    return x.transpose(-1, -2)