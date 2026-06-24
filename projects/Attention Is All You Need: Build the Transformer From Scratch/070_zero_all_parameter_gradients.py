import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    for param in parameter_list:
        if param.grad is not None:
            param.grad.detach_()
            param.grad = None
