import numpy as np

def softmax(logits):
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(logits - max_logits)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)
