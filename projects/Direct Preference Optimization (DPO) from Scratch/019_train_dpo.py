def train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    trained_params = {key: value.copy() for key, value in params.items()}
    history = []
    
    n_pairs = len(pairs)
    
    if isinstance(ref_logprobs, dict) and 'chosen' in ref_logprobs and 'rejected' in ref_logprobs:
        ref_logprobs_list = []
        for i in range(len(ref_logprobs['chosen'])):
            ref_logprobs_list.append({
                'chosen': ref_logprobs['chosen'][i],
                'rejected': ref_logprobs['rejected'][i]
            })
        ref_logprobs = ref_logprobs_list
    
    for step in range(num_steps):
        indices = rng.choice(n_pairs, size=batch_size, replace=(batch_size > n_pairs))
        indices = [int(i) for i in indices]
        
        batch = {}
        keys = ['chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask']
        for key in keys:
            batch[key] = np.array([pairs[i][key] for i in indices])
        
        ref_logprobs_batch = {
            'chosen': np.array([ref_logprobs[i]['chosen'] for i in indices]),
            'rejected': np.array([ref_logprobs[i]['rejected'] for i in indices])
        }
        trained_params, metrics = dpo_train_step(trained_params, batch, ref_logprobs_batch, beta, learning_rate)
        metrics['step'] = step
        history.append(metrics)
    
    return trained_params, history
