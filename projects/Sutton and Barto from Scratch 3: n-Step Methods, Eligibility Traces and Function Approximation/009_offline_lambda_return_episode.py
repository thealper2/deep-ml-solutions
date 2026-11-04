import numpy as np

def offline_lambda_return_episode(values, states, rewards, lam, alpha, gamma=1.0):
    v = np.array(values, dtype=float, copy=True)
    T = len(rewards)
    targets = [0.0] * T

    G = 0.0
    for t in range(T - 1, -1, -1):
        next_v = v[states[t + 1]] if t + 1 < T else 0.0
        G = rewards[t] + gamma * ((1.0 - lam) * next_v + lam * G)
        targets[t] = G

    for t in range(T):
        s = states[t]
        v[s] += alpha * (targets[t] - v[s])

    return v