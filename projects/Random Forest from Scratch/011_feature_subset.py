import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    return rng.choice(num_features, size=num_to_pick, replace=False)
