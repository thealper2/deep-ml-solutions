import numpy as np

def batch_td_prediction(episodes, alpha, theta, gamma=1.0):
    values = np.zeros(7, dtype=float)

    while True:
        increments = np.zeros(7, dtype=float)

        for episode in episodes:
            for state, reward, next_state in episode:
                td_error = reward + gamma * values[next_state] - values[state]
                increments[state] += alpha * td_error

        values += increments

        if np.max(np.abs(increments)) < theta:
            break

    return values