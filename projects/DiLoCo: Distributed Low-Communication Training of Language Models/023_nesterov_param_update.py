def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    if 'momentum' in outer_state:
        momentum = outer_state['momentum']
    else:
        momentum = outer_state
    
    new_params = {}
    for key in params:
        new_params[key] = params[key] - outer_lr * (momentum_coef * momentum[key] + outer_grad[key])
        
    return new_params
