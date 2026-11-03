import numpy as np

def infinite_variance_returns(n_episodes, rng):
    returns = np.zeros(n_episodes, dtype=float)
    ratios = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        n_left = 0

        while True:
            action = int(rng.integers(0, 2))
            if action == 1:
                break

            n_left += 1

            if rng.random() < 0.1:
                returns[i] = 1.0
                ratios[i] = 2.0 ** n_left
                break

    return returns, ratios