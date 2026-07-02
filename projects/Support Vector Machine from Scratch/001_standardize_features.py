import numpy as np

def standardize_features(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std
