import numpy as np

def deserialize_q_table_from_dict(serialized):
    """Rebuild a Q-table (state_key -> np.ndarray shape (9,)) from a plain dict."""
    result = {}
    for key, value in serialized.items():
        result[key] = np.array(value, dtype=np.float64)
        
    return result
