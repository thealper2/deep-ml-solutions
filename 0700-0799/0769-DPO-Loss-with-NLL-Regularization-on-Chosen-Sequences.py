import numpy as np

def dpo_nll_loss(policy_chosen_logps: np.ndarray,
                 policy_rejected_logps: np.ndarray,
                 ref_chosen_logps: np.ndarray,
                 ref_rejected_logps: np.ndarray,
                 chosen_avg_logps: np.ndarray,
                 beta: float,
                 alpha: float) -> float:
    """Compute DPO loss with NLL regularization on chosen sequences."""
    pi_logratios_chosen = policy_chosen_logps - ref_chosen_logps
    pi_logratios_rejected = policy_rejected_logps - ref_rejected_logps

    logits_diff = pi_logratios_chosen - pi_logratios_rejected
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    dpo_losses = -np.log(sigmoid(beta * logits_diff))
    dpo_loss = np.mean(dpo_losses)

    nll_loss = -np.mean(chosen_avg_logps)
    total_loss = dpo_loss + alpha * nll_loss
    return round(total_loss, 4)