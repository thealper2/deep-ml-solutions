import numpy as np

def multilabel_targets(y):
    y = np.asarray(y)
    return np.c_[y >= 7, y % 2 == 1]