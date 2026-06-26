import numpy as np

def mask_illegal_actions_neg_inf(q_values, legal_action_mask):
    """Return a copy of q_values with illegal entries set to -inf."""
    masked_q_values = q_values.copy()
    masked_q_values[~legal_action_mask] = float('-inf')
    return masked_q_values
