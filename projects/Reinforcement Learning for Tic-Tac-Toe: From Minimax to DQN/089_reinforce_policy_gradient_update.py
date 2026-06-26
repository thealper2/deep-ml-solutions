def reinforce_policy_gradient_update(params, episode_cache, returns, adam_state, learning_rate=1e-2):
    states = episode_cache['states']
    actions = episode_cache['actions']
    legal_masks = episode_cache['legal_masks']
    T = len(returns)
    
    grad_accum = {key: np.zeros_like(params[key]) for key in params}
    
    for t in range(T):
        state = states[t:t+1]
        action = actions[t]
        legal_mask = legal_masks[t].astype(bool)
        G_t = returns[t]
        
        q_values, cache = mlp_forward_pass(params, state)
        
        masked_q = mask_illegal_actions_neg_inf(q_values.flatten(), legal_mask)
        
        max_logit = np.max(masked_q)
        exp_logits = np.exp(masked_q - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        dlog = np.zeros_like(probs)
        dlog[action] = 1.0
        dlog -= probs
        
        grad_q = G_t * dlog.reshape(1, -1)
        
        x, z1, h1, q = cache['x'], cache['z1'], cache['h1'], cache['q']
        W2 = params['W2']
        
        dq = grad_q
        
        db2 = np.sum(dq, axis=0)
        dW2 = h1.T @ dq
        
        dh1 = dq @ W2.T
        dz1 = dh1 * (z1 > 0)
        
        db1 = np.sum(dz1, axis=0)
        dW1 = x.T @ dz1
        
        grad_accum['W1'] += dW1
        grad_accum['b1'] += db1
        grad_accum['W2'] += dW2
        grad_accum['b2'] += db2
    
    grad_neg = {key: -grad_accum[key] for key in grad_accum}
    new_params, new_adam_state = adam_update_step(
        params, grad_neg, adam_state, learning_rate
    )
    
    return new_params, new_adam_state
