def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    logits, caches = lenet_forward(xb, params)
    loss = softmax_cross_entropy_forward(logits, yb)
    dlogits = softmax_cross_entropy_backward(logits, yb)
    grads = lenet_backward(dlogits, caches)

    new_params = {}
    new_opt_state = {}
    
    for layer_name in params:
        new_params[layer_name] = {}
        new_opt_state[layer_name] = {}
        grad_mapping = {'W': 'dW', 'b': 'db'}
        
        for param_name in params[layer_name]:
            param = params[layer_name][param_name]
            grad_name = grad_mapping[param_name]
            grad = grads[layer_name][grad_name]
            
            m = opt_state[layer_name][param_name]['m']
            v = opt_state[layer_name][param_name]['v']
            
            new_param, new_m, new_v = adam_step(
                param, grad, m, v, step, lr, beta_one, beta_two, eps
            )
            
            new_params[layer_name][param_name] = new_param
            new_opt_state[layer_name][param_name] = {'m': new_m, 'v': new_v}
    
    return new_params, new_opt_state, loss
