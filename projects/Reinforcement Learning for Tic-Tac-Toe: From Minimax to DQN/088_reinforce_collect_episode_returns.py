import numpy as np

def reinforce_collect_episode_returns(rewards, gamma):
    """Return discounted returns G_t for a REINFORCE episode as a numpy array of shape (T,)."""
    T = len(rewards)
    if T == 0:
        return np.array([])

    returns = np.zeros(T)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running

    return returns
