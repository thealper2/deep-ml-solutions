import numpy as np


def run_access_control(n_steps, alpha, beta, epsilon, rng, n_servers=10, free_prob=0.06):
    q_table = np.zeros((n_servers + 1, 4, 2))
    avg_reward = 0.0
    state = (n_servers, int(rng.integers(0, 4)))

    def choose(s):
        if s[0] == 0:
            return 0
        return epsilon_greedy_tabular(q_table, s, epsilon, rng)

    action = choose(state)
    for _ in range(n_steps):
        next_state, reward = access_control_step(state, action, rng, n_servers, free_prob)
        next_action = choose(next_state)
        q_table, avg_reward = differential_sarsa_update(
            q_table, avg_reward, state, action, reward, next_state, next_action, alpha, beta
        )
        state = next_state
        action = next_action
    return q_table, avg_reward