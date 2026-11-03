import numpy as np

def mc_prediction_blackjack(episodes, first_visit=True):
    values = np.zeros((10, 10, 2), dtype=float)
    counts = np.zeros((10, 10, 2), dtype=float)

    for episode in episodes:
        states = [step[0] for step in episode]
        rewards = [step[2] for step in episode]

        returns = episode_returns(rewards, gamma=1.0)

        visited = set()

        for state, G in zip(states, returns):
            player_sum, dealer_showing, usable_ace = state

            if first_visit and state in visited:
                continue

            visited.add(state)

            idx = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace)
            )

            counts[idx] += 1
            values[idx] += (G - values[idx]) / counts[idx]

    return values, counts