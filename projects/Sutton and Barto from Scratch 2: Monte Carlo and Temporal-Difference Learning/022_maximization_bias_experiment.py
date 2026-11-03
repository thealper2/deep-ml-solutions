import numpy as np

def maximization_bias_experiment(n_runs, n_episodes, alpha, epsilon, rng, double=True):
    left_counts = np.zeros(n_episodes, dtype=float)

    for _ in range(n_runs):
        q1 = np.zeros((2, 10), dtype=float)
        q2 = np.zeros((2, 10), dtype=float)

        for episode_idx in range(n_episodes):
            state = 0
            done = False

            while not done:
                if state == 0:
                    valid_actions = 2
                else:
                    valid_actions = 10

                if double:
                    action_values = q1[state, :valid_actions] + q2[state, :valid_actions]
                else:
                    action_values = q1[state, :valid_actions]

                action = epsilon_soft_action(action_values, epsilon, rng)

                if state == 0:
                    if action == 0:
                        left_counts[episode_idx] += 1
                        next_state = 1
                        reward = 0.0
                        done = False
                    else:
                        next_state = 2
                        reward = 0.0
                        done = True
                else:
                    next_state = 2
                    reward = float(rng.normal(-0.1, 1.0))
                    done = True

                if double:
                    q1, q2 = double_q_update(
                        q1, q2, state, action, reward,
                        next_state, done, alpha, rng
                    )
                else:
                    if done:
                        target = reward
                    else:
                        target = reward + np.max(q1[next_state])

                    q1[state, action] += alpha * (
                        target - q1[state, action]
                    )

                state = next_state

    return left_counts / n_runs