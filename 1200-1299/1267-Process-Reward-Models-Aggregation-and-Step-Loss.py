import numpy as np

def aggregate_process_rewards(step_rewards, method="mean"):
    """
    Aggregate step-level rewards. method in {mean, min, prod, last}.
    Round to 6 decimals. Empty -> 0.0.
    """
    if not step_rewards:
        return 0.0

    step_rewards = np.array(step_rewards)

    if method == "mean":
        result = np.mean(step_rewards)
    elif method == "min":
        result = np.min(step_rewards)
    elif method == "prod":
        result = np.prod(step_rewards)
    elif method == "last":
        result = step_rewards[-1]
    else:
        raise ValueError(f"Unknown method: {method}")

    return round(float(result), 6)

def process_reward_loss(step_logits, step_labels):
    """
    BCE-with-logits PRM loss: mean(softplus(z) - y*z), 6 decimals.
    """
    step_logits = np.array(step_logits)
    step_labels = np.array(step_labels)
    softplus = np.logaddexp(0, step_logits)
    losses = softplus - step_labels * step_logits
    return round(float(np.mean(losses)), 6)

def outcome_and_process_scores(step_rewards, outcome_reward, method="mean"):
    """
    Return (process_score, outcome_reward, 0.5*p + 0.5*o), 6 decimals.
    """
    process_score = aggregate_process_rewards(step_rewards, method=method)
    outcome_reward = round(float(outcome_reward), 6)
    combined = round(0.5 * process_score + 0.5 * outcome_reward, 6)
    return process_score, outcome_reward, combined
