import numpy as np

def gradient_mc_episode(weights, states, rewards, n_states, n_groups, alpha, gamma=1.0):
    w = np.array(weights, dtype=float, copy=True)
    T = len(rewards)

    returns = [0.0] * T
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    for t in range(T):
        x = state_aggregation_features(states[t], n_states, n_groups)
        w += alpha * (returns[t] - w @ x) * x

    return w