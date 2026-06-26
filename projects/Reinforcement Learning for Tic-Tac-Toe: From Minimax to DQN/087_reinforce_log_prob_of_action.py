import numpy as np

def reinforce_log_prob_of_action(logits, legal_action_mask, action):
    """Return (log_prob_of_action, full_prob_vector) under a softmax policy with illegal cells masked out."""
    masked_logits = mask_illegal_actions_neg_inf(logits, legal_action_mask)
    max_logit = np.max(masked_logits)
    exp_logits = np.exp(masked_logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    log_prob = np.log(probs[action] + 1e-12)
    return log_prob, probs
