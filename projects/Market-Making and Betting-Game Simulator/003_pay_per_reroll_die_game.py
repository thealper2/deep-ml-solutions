import numpy as np

def pay_per_reroll_die_game(sides, reroll_cost):
    best_threshold = 1
    best_value = -float("inf")

    for threshold in range(1, sides + 1):
        keep_values = np.arange(threshold, sides + 1, dtype=float)
        keep_prob = (sides - threshold + 1) / sides

        if keep_prob == 0:
            continue

        keep_mean = float(np.mean(keep_values))
        value = keep_mean - ((1 - keep_prob) / keep_prob) * reroll_cost

        if value > best_value + 1e-12:
            best_value = value
            best_threshold = threshold

    return {
        "threshold": best_threshold,
        "value": best_value
    }
