import numpy as np

def simulate_dice_sum(num_rolls, seed=0):
    rng = np.random.RandomState(seed)

    roll1 = rng.randint(1, 7, size=num_rolls)
    roll2 = rng.randint(1, 7, size=num_rolls)
    sums = roll1 + roll2

    empirical = np.zeros(11)
    for s in range(2, 13):
        empirical[s - 2] = np.sum(sums == s) / num_rolls

    theoretical = np.zeros(11)
    for s in range(2, 13):
        count = 0
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                if d1 + d2 == s:
                    count += 1

        theoretical[s - 2] = count / 36

    return empirical, theoretical
