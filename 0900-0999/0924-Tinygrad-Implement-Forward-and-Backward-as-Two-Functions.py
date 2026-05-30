from tinygrad import Tensor

def relu_forward(x):
    return x.maximum(0)

def relu_backward(x, grad_output):
    return grad_output * (x > 0)