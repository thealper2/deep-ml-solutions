import numpy as np

def optimistic_greedy_bandit(
    true_rewards: list,
    initial_q: float,
    n_steps: int,
    step_size: float
) -> tuple:
    """
    Simulate a greedy bandit agent with optimistic initialization.
    
    Args:
        true_rewards: List of true deterministic rewards for each arm
        initial_q: Optimistic initial Q-value for all arms
        n_steps: Number of steps to simulate
        step_size: Constant step-size (alpha) for Q-value updates
    
    Returns:
        Tuple of (Q_values, action_counts) where Q_values is a list of
        floats rounded to 4 decimal places, and action_counts is a list of ints.
    """
    n_arms = len(true_rewards)

    q_values = [initial_q] * n_arms
    counts = [0] * n_arms

    for _ in range(n_steps):
        selected_arm = max(range(n_arms), key=lambda i: q_values[i])
        reward = true_rewards[selected_arm]
        q_values[selected_arm] += step_size * (reward - q_values[selected_arm])
        counts[selected_arm] += 1

    final_q_values = [round(q, 4) for q in q_values]
    return final_q_values, counts