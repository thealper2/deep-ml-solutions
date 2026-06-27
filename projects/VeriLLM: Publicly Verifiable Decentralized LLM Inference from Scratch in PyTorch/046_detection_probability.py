import math

def detection_probability(num_steps, num_corrupted, k):
    if k == 0 or num_corrupted == 0:
        return 0.0

    if num_corrupted >= num_steps:
        return 1.0

    if k >= num_steps:
        return 1.0 if num_corrupted > 0 else 0.0

    clean = num_steps - num_corrupted
    if k > clean:
        return 1.0

    miss_prob = math.comb(clean, k) / math.comb(num_steps, k)
    return 1.0 - miss_prob
