import numpy as np

def td_lambda_episode(values, states, rewards, lam, alpha, gamma, mode):
    v = np.array(values, dtype=float, copy=True)
    traces = np.zeros_like(v)
    T = len(rewards)

    for t in range(T):
        s = states[t]
        next_v = v[states[t + 1]] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - v[s]
        traces = update_traces(traces, s, lam, gamma, mode)
        v = v + alpha * delta * traces

    return v