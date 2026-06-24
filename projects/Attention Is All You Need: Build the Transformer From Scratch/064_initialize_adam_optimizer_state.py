import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    m = []
    v = []
    for param in parameter_list:
        m.append(torch.zeros_like(param, requires_grad=False))
        v.append(torch.zeros_like(param, requires_grad=False))
        
    return {'m': m, 'v': v, 't': 0}
