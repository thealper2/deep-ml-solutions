import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    return d_scaled_scores / np.sqrt(d_head)
