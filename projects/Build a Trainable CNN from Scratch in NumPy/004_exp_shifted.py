import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    max_values = row_max(logits)
    exp_values = np.exp(logits - max_values)
    return exp_values
