import numpy as np

def mc_control_epsilon_soft(n_episodes, epsilon, rng):
    q_table = np.zeros((10, 10, 2, 2), dtype=float)
    counts = np.zeros((10, 10, 2, 2), dtype=float)

    for _ in range(n_episodes):
        def policy_fn(state, r):
            player_sum, dealer_showing, usable_ace = state

            idx = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace)
            )

            return epsilon_soft_action(q_table[idx], epsilon, r)

        episode = play_blackjack_episode(policy_fn, rng)

        states = [step[0] for step in episode]
        actions = [step[1] for step in episode]
        rewards = [step[2] for step in episode]

        returns = episode_returns(rewards)

        visited = set()

        for state, action, G in zip(states, actions, returns):
            player_sum, dealer_showing, usable_ace = state

            idx = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace),
                action
            )

            if idx in visited:
                continue

            visited.add(idx)

            counts[idx] += 1
            q_table[idx] += (G - q_table[idx]) / counts[idx]

    policy = np.argmax(q_table, axis=3)

    return q_table, policy