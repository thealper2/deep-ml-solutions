import numpy as np

def tile_epsilon_greedy(weights, active_tiles, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, weights.shape[0]))
    
    q = weights[:, active_tiles].sum(axis=1)
    return int(np.argmax(q))