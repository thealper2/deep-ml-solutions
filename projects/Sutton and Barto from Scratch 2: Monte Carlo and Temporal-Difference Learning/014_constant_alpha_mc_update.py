import numpy as np

def constant_alpha_mc_update(values, transitions, alpha, gamma=1.0):
    updated_values = np.array(values, dtype=float, copy=True)

    rewards = [reward for state, reward, next_state in transitions]
    returns = episode_returns(rewards, gamma)

    for (state, reward, next_state), G in zip(transitions, returns):
        updated_values[state] += alpha * (G - updated_values[state])

    return updated_values