import numpy as np

def double_q_update(q1, q2, state, action, reward, next_state, done, alpha, rng):
    q1 = np.array(q1, dtype=float, copy=True)
    q2 = np.array(q2, dtype=float, copy=True)

    if rng.random() < 0.5:
        if done:
            target = reward
        else:
            best_action = np.argmax(q1[next_state])
            target = reward + q2[next_state, best_action]

        q1[state, action] += alpha * (
            target - q1[state, action]
        )
    else:
        if done:
            target = reward
        else:
            best_action = np.argmax(q2[next_state])
            target = reward + q1[next_state, best_action]

        q2[state, action] += alpha * (
            target - q2[state, action]
        )

    return q1, q2