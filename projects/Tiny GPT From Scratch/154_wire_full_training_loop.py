def wire_full_training_loop(params, train_ids, val_ids, block_size, batch_size, n_steps, lr, betas, eps):
    """Run the full GPT training loop for n_steps and return (updated_params, history)."""
    rng = np.random.default_rng(0)
    history = []

    m, v = initialize_adam_moments(params)
    t = initialize_adam_step_counter()
    
    beta1, beta2 = betas
    
    for step in range(n_steps):
        X_batch, Y_batch = get_batch(train_ids, block_size, batch_size, rng)
        
        logits, caches = full_model_forward(X_batch, params)
        
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = Y_batch.reshape(-1)
        
        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(targets_flat)), targets_flat] = 1.0
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-15), axis=1))
        
        d_logits_flat = (probs - one_hot) / (B * T)
        d_logits = d_logits_flat.reshape(B, T, V)
        
        grads = full_model_backward(d_logits, caches, params)
        
        t += 1
        
        def update_recursive(param_tree, grad_tree, m_tree, v_tree, t, lr, beta1, beta2, eps):
            if isinstance(param_tree, dict):
                for key in param_tree:
                    param_tree[key], grad_tree[key], m_tree[key], v_tree[key] = update_recursive(
                        param_tree[key], grad_tree[key], m_tree[key], v_tree[key], t, lr, beta1, beta2, eps
                    )
                return param_tree, grad_tree, m_tree, v_tree
            elif isinstance(param_tree, list):
                for i in range(len(param_tree)):
                    param_tree[i], grad_tree[i], m_tree[i], v_tree[i] = update_recursive(
                        param_tree[i], grad_tree[i], m_tree[i], v_tree[i], t, lr, beta1, beta2, eps
                    )
                return param_tree, grad_tree, m_tree, v_tree
            else:
                m_new = beta1 * m_tree + (1 - beta1) * grad_tree
                v_new = beta2 * v_tree + (1 - beta2) * (grad_tree ** 2)
                
                m_hat = m_new / (1 - beta1 ** t)
                v_hat = v_new / (1 - beta2 ** t)
                
                param_new = param_tree - lr * m_hat / (np.sqrt(v_hat) + eps)
                return param_new, grad_tree, m_new, v_new
        
        params, grads, m, v = update_recursive(params, grads, m, v, t, lr, beta1, beta2, eps)
        
        history.append({'step': step, 'train_loss': loss})
    
    return params, history
