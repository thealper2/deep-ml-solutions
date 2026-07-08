import numpy as np

def decoupled_weight_decay(params, lr, weight_decay):
    new_params = {}
    decay_factor = 1.0 - lr * weight_decay
    for key in params:
        new_params[key] = params[key] * decay_factor
    
    return new_params
