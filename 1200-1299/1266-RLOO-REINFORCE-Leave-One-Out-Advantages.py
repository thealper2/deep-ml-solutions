import numpy as np

def rloo_advantages(rewards):
    """
    Leave-one-out advantages for a group of rewards.
    G==1 -> [0.0]. Round to 6 decimals.
    """
    G = len(rewards)
    if G <= 1:
        return [0.0]

    total = np.sum(rewards)
    baselines = (total - rewards) / (G - 1)
    advantages = rewards - baselines
    return [round(float(a), 6) for a in advantages]

def rloo_loss(log_probs, rewards, kl_log_ratios=None, beta=0.0):
    """
    RLOO policy-gradient loss (minimization form).
    Optionally apply r <- r - beta * kl_log_ratios first.
    Return mean loss rounded to 6 decimals.
    """
    log_probs = np.array(log_probs)
    rewards = np.array(rewards)
    
    if kl_log_ratios is not None:
        kl_log_ratios = np.array(kl_log_ratios)
        rewards -= beta * kl_log_ratios

    advantages = rloo_advantages(rewards)
    loss = -np.mean(advantages * log_probs)
    return round(float(loss), 6)
