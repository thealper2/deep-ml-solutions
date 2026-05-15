import torch

def linear_backward(grad_output, x, W):
    grad_input = grad_output @ W
    grad_W = grad_output.T @ x
    grad_b = grad_output.sum(dim=0)
    return grad_input, grad_W, grad_b