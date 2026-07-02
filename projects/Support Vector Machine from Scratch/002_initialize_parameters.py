import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    return {
        'w': np.zeros(n_features),
        'b': 0.0,
    }
