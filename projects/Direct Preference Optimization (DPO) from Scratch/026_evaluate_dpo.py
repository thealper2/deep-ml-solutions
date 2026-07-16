def evaluate_dpo(params, pairs, ref_logprobs, beta):
    policy_logprob_chosen = []
    policy_logprob_rejected = []
    ref_logprob_chosen = []
    ref_logprob_rejected = []
    
    for i, pair in enumerate(pairs):
        chosen_ids = np.array(pair['chosen_ids']).reshape(1, -1)
        chosen_mask = np.array(pair['chosen_mask']).reshape(1, -1)
        rejected_ids = np.array(pair['rejected_ids']).reshape(1, -1)
        rejected_mask = np.array(pair['rejected_mask']).reshape(1, -1)
        
        pc = policy_sequence_logprob(params, chosen_ids, chosen_mask)
        pr = policy_sequence_logprob(params, rejected_ids, rejected_mask)
        
        policy_logprob_chosen.append(float(pc[0]) if hasattr(pc, '__len__') else float(pc))
        policy_logprob_rejected.append(float(pr[0]) if hasattr(pr, '__len__') else float(pr))
        
        if isinstance(ref_logprobs[i], dict):
            ref_logprob_chosen.append(float(ref_logprobs[i]['chosen']))
            ref_logprob_rejected.append(float(ref_logprobs[i]['rejected']))
        else:
            ref_logprob_chosen.append(float(ref_logprobs[i][0]) if hasattr(ref_logprobs[i], '__len__') else float(ref_logprobs[i]))
            ref_logprob_rejected.append(float(ref_logprobs[i][1]) if hasattr(ref_logprobs[i], '__len__') else 0.0)
    
    policy_logprob_chosen = np.array(policy_logprob_chosen)
    policy_logprob_rejected = np.array(policy_logprob_rejected)
    ref_logprob_chosen = np.array(ref_logprob_chosen)
    ref_logprob_rejected = np.array(ref_logprob_rejected)
    
    dpo_loss_value = dpo_loss(policy_logprob_chosen, policy_logprob_rejected,
                              ref_logprob_chosen, ref_logprob_rejected, beta)
    
    pref_acc = preference_accuracy(policy_logprob_chosen, policy_logprob_rejected,
                                   ref_logprob_chosen, ref_logprob_rejected, beta)
    
    all_policy = np.concatenate([policy_logprob_chosen, policy_logprob_rejected])
    all_ref = np.concatenate([ref_logprob_chosen, ref_logprob_rejected])
    kl = kl_to_reference(all_policy, all_ref)
    
    margin_stats = reward_margin_stats(policy_logprob_chosen, policy_logprob_rejected,
                                       ref_logprob_chosen, ref_logprob_rejected, beta)
    
    return {
        'dpo_loss': float(dpo_loss_value),
        'preference_accuracy': float(pref_acc),
        'kl_to_reference': float(kl),
        'mean_margin': float(margin_stats['mean_margin']),
        'std_margin': float(margin_stats['std_margin']),
        'frac_positive': float(margin_stats['frac_positive'])
    }
