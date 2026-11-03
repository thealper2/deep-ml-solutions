import numpy as np

def sarsa_cliff(n_episodes, alpha, epsilon, rng):
    q_table = np.zeros((48, 4), dtype=float)
    episode_rewards = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        state = 36
        action = epsilon_soft_action(q_table[state], epsilon, rng)
        done = False

        while not done:
            next_state, reward, done = cliff_step(state, action)
            episode_rewards[i] += reward

            if done:
                target = reward
            else:
                next_action = epsilon_soft_action(
                    q_table[next_state], epsilon, rng
                )
                target = reward + q_table[next_state, next_action]

            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            if not done:
                state = next_state
                action = next_action

    return q_table, episode_rewards