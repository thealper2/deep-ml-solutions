import numpy as np

def state_aggregation_features(state, n_states, n_groups):
    x = np.zeros(n_groups, dtype=np.float64)
    x[state * n_groups // n_states] = 1.0
    return x