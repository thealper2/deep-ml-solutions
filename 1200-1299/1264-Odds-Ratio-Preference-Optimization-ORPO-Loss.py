import numpy as np

def orpo_loss(chosen_logps, rejected_logps, lambda_coef=0.1):
    """
    Odds Ratio Preference Optimization (ORPO) loss.

    Args:
        chosen_logps: sequence log πθ(y_w) (negative floats)
        rejected_logps: sequence log πθ(y_l)
        lambda_coef: weight on the odds-ratio term

    Returns:
        Mean ORPO loss, rounded to 6 decimals.
    """
    chosen_logps = np.array(chosen_logps)
    rejected_logps = np.array(rejected_logps)
    chosen_logps = np.minimum(chosen_logps, -1e-7)
    rejected_logps = np.minimum(rejected_logps, -1e-7)
    odds_chosen = chosen_logps - np.log1p(-np.exp(chosen_logps))
    odds_rejected = rejected_logps - np.log1p(-np.exp(rejected_logps))
    odds_diff = odds_chosen - odds_rejected
    L_OR = np.logaddexp(0, -odds_diff)
    L_SFT = -chosen_logps
    losses = L_SFT + lambda_coef * L_OR
    return round(float(np.mean(losses)), 6)
