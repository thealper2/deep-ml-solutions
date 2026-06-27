import numpy as np

def weighted_value_sum(attn_weights, values):
    return np.matmul(attn_weights, values)
