import numpy as np

def td0_update(values, transitions, alpha, gamma=1.0):
    updated_values = np.array(values, dtype=float, copy=True)

    for state, reward, next_state in transitions:
        td_target = reward + gamma * updated_values[next_state]
        td_error = td_target - updated_values[state]
        updated_values[state] += alpha * td_error

    return updated_values