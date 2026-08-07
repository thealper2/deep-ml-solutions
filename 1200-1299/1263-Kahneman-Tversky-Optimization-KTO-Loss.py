import numpy as np

def kto_loss(policy_logps, ref_logps, labels, beta, kl_mean=None,
             desirable_weight=1.0, undesirable_weight=1.0):
    """
    Kahneman-Tversky Optimization (KTO) loss.

    Args:
        policy_logps: log πθ(y|x) per example
        ref_logps: log πref(y|x) per example
        labels: 1 = desirable, 0 = undesirable
        beta: temperature on the implicit reward
        kl_mean: optional detached KL estimate z0; default = mean reward
        desirable_weight: λ_d
        undesirable_weight: λ_u

    Returns:
        Mean KTO loss, rounded to 6 decimals.
    """
    policy_logps = np.array(policy_logps)
    ref_logps = np.array(ref_logps)
    labels = np.array(labels)

    rewards = beta * (policy_logps - ref_logps)

    if kl_mean is None:
        z0 = np.mean(rewards)
    else:
        z0 = kl_mean

    losses = np.zeros_like(labels, dtype=float)

    for i, label in enumerate(labels):
        if label == 1:
            loss = desirable_weight * (1.0 / (1.0 + np.exp(-(z0 - rewards[i]))))
            losses[i] = loss
        else:
            loss = undesirable_weight * (1.0 / (1.0 + np.exp(-(rewards[i] - z0))))
            losses[i] = loss
    
    return round(float(np.mean(losses)), 6)
