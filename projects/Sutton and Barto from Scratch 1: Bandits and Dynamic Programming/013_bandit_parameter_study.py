def bandit_parameter_study(n_runs, n_steps, seed, settings):
    k = 10
    results = {}
    
    for setting in settings:
        method = setting['method']
        param = setting['param']
        nonstationary = setting.get('nonstationary', False)
        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"
        
        total_rewards = np.zeros(n_steps)
        
        for run in range(n_runs):
            run_seed = seed + run
            if nonstationary:
                true_values = np.zeros(k)
            else:
                true_values = create_bandit_testbed(k, run_seed)
            
            if method == 'epsilon_greedy':
                q_values = np.zeros(k)
                action_counts = np.zeros(k, dtype=int)
                eps = param
            elif method == 'constant_step':
                q_values = np.zeros(k)
                action_counts = np.zeros(k, dtype=int)
                alpha = param
                eps = 0.1
            elif method == 'optimistic':
                q_values = optimistic_initialization(k, param)
                action_counts = np.zeros(k, dtype=int)
                alpha = 0.1
                eps = 0.0
            elif method == 'ucb':
                q_values = np.zeros(k)
                action_counts = np.zeros(k, dtype=int)
                c = param
            elif method == 'gradient':
                preferences = np.zeros(k)
                avg_reward = 0.0
                alpha = param
                step_count = 0
            else:
                raise ValueError(f"Unknown method: {method}")
            
            rng = np.random.default_rng(run_seed)
            rewards = np.zeros(n_steps)
            
            for step in range(n_steps):
                t = step + 1
                
                if method == 'epsilon_greedy' or method == 'constant_step':
                    if rng.random() < eps:
                        action = rng.integers(0, k)
                    else:
                        action = int(np.argmax(q_values))
                elif method == 'optimistic':
                    action = int(np.argmax(q_values))
                elif method == 'ucb':
                    action = ucb_action_select(q_values, action_counts, t, c)
                elif method == 'gradient':
                    exp_pref = np.exp(preferences - np.max(preferences))
                    probs = exp_pref / np.sum(exp_pref)
                    action = rng.choice(k, p=probs)
                
                reward = rng.normal(loc=true_values[action], scale=1.0)
                rewards[step] = reward
                
                if nonstationary:
                    true_values = apply_random_walk_drift(true_values, 0.01, rng)
                
                if method == 'epsilon_greedy':
                    q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
                elif method == 'constant_step':
                    q_values = constant_step_size_update(q_values, action, reward, alpha)
                    action_counts[action] += 1
                elif method == 'optimistic':
                    q_values = constant_step_size_update(q_values, action, reward, alpha)
                    action_counts[action] += 1
                elif method == 'ucb':
                    q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
                elif method == 'gradient':
                    step_count += 1
                    avg_reward += (reward - avg_reward) / step_count
                    preferences = gradient_bandit_update(preferences, action, reward, avg_reward, alpha)
            
            total_rewards += rewards
        
        results[label] = np.mean(total_rewards[-1] / n_runs)
    
    return results
