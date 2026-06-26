import numpy as np

def adam_update_step(params, grads, adam_state, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    if 't' not in adam_state:
        adam_state['t'] = 0
        adam_state['m'] = {}
        adam_state['v'] = {}
        for key in params:
            adam_state['m'][key] = np.zeros_like(params[key])
            adam_state['v'][key] = np.zeros_like(params[key])
    
    adam_state['t'] += 1
    t = adam_state['t']
    
    new_params = {}
    for key in params:
        grad = grads[key]
        m = adam_state['m'][key]
        v = adam_state['v'][key]
        
        m_new = beta1 * m + (1 - beta1) * grad
        adam_state['m'][key] = m_new
        
        v_new = beta2 * v + (1 - beta2) * (grad ** 2)
        adam_state['v'][key] = v_new
        
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        
        new_params[key] = params[key] - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    
    return new_params, adam_state
