import numpy as np

def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_lengths, rejected_lengths, beta=2.0, gamma=1.0):
    """
    Simple Preference Optimization (SimPO) loss.

    Args:
        policy_chosen_logps: summed log πθ(y_w)
        policy_rejected_logps: summed log πθ(y_l)
        chosen_lengths: |y_w| token counts
        rejected_lengths: |y_l| token counts
        beta: scaling on the length-normalized gap
        gamma: target reward margin

    Returns:
        Mean SimPO loss, rounded to 6 decimals.
    """
    s_chosen = np.array(policy_chosen_logps) / np.array(chosen_lengths)
    s_rejected = np.array(policy_rejected_logps) / np.array(rejected_lengths)
    gap = s_chosen - s_rejected
    logits = beta * gap - gamma
    losses = np.logaddexp(0, -logits)
    return round(float(np.mean(losses)), 6)
