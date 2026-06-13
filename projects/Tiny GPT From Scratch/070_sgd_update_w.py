import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    new_w = w - (learning_rate * dw)
    return new_w
