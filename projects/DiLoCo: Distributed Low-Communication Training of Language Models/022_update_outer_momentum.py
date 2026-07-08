import numpy as np

def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    """Update Nesterov momentum buffer: m <- momentum_coef * m + outer_grad."""
    for key in outer_state['momentum']:
        outer_state['momentum'][key] = momentum_coef * outer_state['momentum'][key] + outer_grad[key]
    
    return outer_state
