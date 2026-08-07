import numpy as np

def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    """
    Identity Preference Optimization (IPO) loss.

    Args:
        policy_chosen_logps: log πθ(y_w) per example
        policy_rejected_logps: log πθ(y_l) per example
        ref_chosen_logps: log πref(y_w) per example
        ref_rejected_logps: log πref(y_l) per example
        beta: KL temperature (target gap is 1/(2β))

    Returns:
        Mean IPO loss, rounded to 6 decimals.
    """
    pc = np.array(policy_chosen_logps)
    pr = np.array(policy_rejected_logps)
    rc = np.array(ref_chosen_logps)
    rr = np.array(ref_rejected_logps)

    h = (pc - rc) - (pr - rr)
    target = 1.0 / (2.0 * beta)
    losses = np.mean((h - target) ** 2)
    return round(float(losses), 6)
