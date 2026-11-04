import numpy as np


def differential_sarsa_update(q_table, avg_reward, state, action, reward, next_state, next_action, alpha, beta):
    q = np.array(q_table, dtype=float, copy=True)
    delta = (
        reward
        - float(avg_reward)
        + float(q[next_state][next_action])
        - float(q[state][action])
    )
    new_avg = float(avg_reward) + beta * delta
    q[state][action] += alpha * delta
    return q, new_avg