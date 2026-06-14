import numpy as np

def append_token_to_sequence(context_ids, token_id):
    """Append token_id as a new final column to context_ids of shape (1, T)."""
    return np.concatenate([context_ids, np.array([[token_id]])], axis=1)
