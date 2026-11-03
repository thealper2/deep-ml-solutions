import numpy as np

def episode_returns(rewards, gamma=1.0):
    rewards = np.asarray(rewards, dtype=float)
    returns = np.zeros(len(rewards), dtype=float)

    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns