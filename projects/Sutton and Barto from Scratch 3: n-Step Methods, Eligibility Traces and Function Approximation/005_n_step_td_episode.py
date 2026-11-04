import numpy as np

def n_step_td_episode(values, states, rewards, n, alpha, gamma=1.0):
    v = np.array(values, dtype=float, copy=True)
    T = len(rewards)
    for t in range(T):
        G = n_step_return(rewards, states, t, n, v, gamma)
        s = states[t]
        v[s] += alpha * (G - v[s])

    return v