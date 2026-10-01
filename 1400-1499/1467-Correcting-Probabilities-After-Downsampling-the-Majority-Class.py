import numpy as np


def downsample_negatives(y, rate, rng):
    y = np.asarray(y)
    keep = y == 1
    negatives = y == 0
    draws = rng.random(len(y))
    keep = keep | (negatives & (draws < rate))
    return keep

def corrected_probability(p_sampled, rate):
    p_sampled = np.asarray(p_sampled, dtype=float)
    return rate * p_sampled / (rate * p_sampled + 1 - p_sampled)

def logit_shift(rate):
    return float(np.log(rate))


def sampled_probability(p, rate):
    p = np.asarray(p, dtype=float)
    return p / (p + (1 - p) * rate)

def calibration_gap(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(p) - np.mean(y))
