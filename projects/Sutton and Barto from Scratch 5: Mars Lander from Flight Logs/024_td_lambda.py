import numpy as np


def td_lambda(X, X_next, rewards, dones, alpha, lam, true_online=False, gamma=1.0):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    w = np.zeros(d, dtype=float)
    z = np.zeros(d, dtype=float)
    v_old = 0.0

    for t in range(T):
        x = X[t]
        xn = X_next[t]
        V = float(w @ x)
        V_next = float(w @ xn)
        delta = rewards[t] + gamma * V_next - V

        if true_online:
            z = gamma * lam * z + (1.0 - alpha * gamma * lam * float(z @ x)) * x
            w = w + alpha * (delta + V - v_old) * z - alpha * (V - v_old) * x
            v_old = V_next
        else:
            z = gamma * lam * z + x
            w = w + alpha * delta * z

        if dones[t]:
            z = np.zeros(d, dtype=float)
            v_old = 0.0

    return w