import numpy as np

def off_policy_blackjack_returns(n_episodes, rng):
    returns = np.zeros(n_episodes, dtype=float)
    ratios = np.zeros(n_episodes, dtype=float)

    def behaviour_policy(state, r):
        return int(r.integers(0, 2))

    for i in range(n_episodes):
        dealer_cards = [2, draw_card(rng)]

        episode = play_blackjack_episode(
            behaviour_policy,
            rng,
            player_cards=[1, 2],
            dealer_cards=dealer_cards
        )

        returns[i] = episode[-1][2]

        agrees = all(
            action == (0 if state[0] >= 20 else 1)
            for state, action, reward in episode
        )

        if agrees:
            ratios[i] = 2.0 ** len(episode)
        else:
            ratios[i] = 0.0

    return returns, ratios