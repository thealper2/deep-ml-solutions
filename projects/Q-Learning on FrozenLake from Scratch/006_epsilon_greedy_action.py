import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    else:
        return greedy_action(q_table, state)
