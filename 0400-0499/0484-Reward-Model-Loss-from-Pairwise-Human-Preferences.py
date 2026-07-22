import numpy as np

def reward_model_loss(chosen_rewards: list, rejected_rewards: list, margin: float = 0.0) -> dict:
    """
    Compute reward model loss from pairwise human preferences.
    
    Args:
        chosen_rewards: Reward scores for preferred responses
        rejected_rewards: Reward scores for non-preferred responses
        margin: Minimum desired gap between chosen and rejected scores
    
    Returns:
        Dictionary with 'loss' (float) and 'accuracy' (float)
    """
    chosen = np.array(chosen_rewards)
    rejected = np.array(rejected_rewards)
    diff = chosen - rejected - margin
    loss = np.mean(np.log(1 + np.exp(-diff)))
    accuracy = np.mean(diff > 0)
    return {'loss': round(float(loss), 4), 'accuracy': round(float(accuracy), 4)}
