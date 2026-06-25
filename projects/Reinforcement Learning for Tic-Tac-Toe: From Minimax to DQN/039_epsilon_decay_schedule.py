import numpy as np

def epsilon_decay_schedule(initial_epsilon, episode_index, min_epsilon, decay_rate):
    """Return the decayed epsilon for the given episode, clipped to min_epsilon."""
    epsilon = initial_epsilon * np.exp(-decay_rate * episode_index)
    return max(min_epsilon, epsilon)
