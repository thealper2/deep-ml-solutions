import numpy as np

def q_learning_cliff(n_episodes, alpha, epsilon, rng):
    q_table = np.zeros((48, 4), dtype=float)
    episode_rewards = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        state = 36
        done = False

        while not done:
            action = epsilon_soft_action(q_table[state], epsilon, rng)
            next_state, reward, done = cliff_step(state, action)

            episode_rewards[i] += reward

            if done:
                target = reward
            else:
                target = reward + np.max(q_table[next_state])

            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            state = next_state

    return q_table, episode_rewards