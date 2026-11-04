import numpy as np


def q_learning_update(q_table, state, action, reward, next_state, alpha, gamma):
    q = np.array(q_table, dtype=float, copy=True)
    target = reward + gamma * float(np.max(q[next_state]))
    current = float(q[state][action])
    q[state][action] = current + alpha * (target - current)
    return q