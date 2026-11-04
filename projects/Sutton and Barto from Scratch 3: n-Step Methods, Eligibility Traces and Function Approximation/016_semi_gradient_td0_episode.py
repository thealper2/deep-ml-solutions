import numpy as np

def semi_gradient_td0_episode(weights, states, rewards, n_states, n_groups, alpha, gamma=1.0):
    w = np.array(weights, dtype=float, copy=True)
    T = len(rewards)

    for t in range(T):
        x = state_aggregation_features(states[t], n_states, n_groups)
        if t + 1 < T:
            x_next = state_aggregation_features(states[t + 1], n_states, n_groups)
            next_value = w @ x_next
        else:
            next_value = 0.0
        delta = rewards[t] + gamma * next_value - w @ x
        w = w + alpha * delta * x

    return w