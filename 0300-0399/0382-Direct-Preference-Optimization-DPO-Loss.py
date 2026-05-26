import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    
    Args:
        log_probs_chosen_policy: Log-probs of chosen responses under policy
        log_probs_rejected_policy: Log-probs of rejected responses under policy
        log_probs_chosen_ref: Log-probs of chosen responses under reference model
        log_probs_rejected_ref: Log-probs of rejected responses under reference model
        beta: Temperature parameter for KL constraint strength
    
    Returns:
        Dictionary with 'loss', 'chosen_rewards', and 'rejected_rewards'
    """
    log_probs_chosen_policy = np.array(log_probs_chosen_policy)
    log_probs_rejected_policy = np.array(log_probs_rejected_policy)
    log_probs_chosen_ref = np.array(log_probs_chosen_ref)
    log_probs_rejected_ref = np.array(log_probs_rejected_ref)

    pi_log_ratios = log_probs_chosen_policy - log_probs_rejected_policy
    ref_log_ratios = log_probs_chosen_ref - log_probs_rejected_ref
    logits = beta * (pi_log_ratios - ref_log_ratios)
    loss = -np.mean(np.log(1 / (1 + np.exp(-logits))))
    
    chosen_rewards = beta * (log_probs_chosen_policy - log_probs_chosen_ref)
    rejected_rewards = beta * (log_probs_rejected_policy - log_probs_rejected_ref)

    return {
        'loss': round(loss, 4),
        'chosen_rewards': np.round(chosen_rewards, 4).tolist(),
        'rejected_rewards': np.round(rejected_rewards, 4).tolist(),
    }