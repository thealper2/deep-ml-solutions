import numpy as np


def landing_rate(kind, n_flights, rng, start=START):
    logs = fly(kind, n_flights, rng, start)
    return float(logs["rewards"][logs["dones"]].mean())