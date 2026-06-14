import numpy as np

def initialize_adam_moments(model_params):
    """Allocate zeroed Adam first- and second-moment buffers matching model_params."""
    if isinstance(model_params, dict):
        m = {}
        v = {}
        for key, value in model_params.items():
            m[key], v[key] = initialize_adam_moments(value)
        return m, v
    elif isinstance(model_params, list):
        m = []
        v = []
        for item in model_params:
            m_item, v_item = initialize_adam_moments(item)
            m.append(m_item)
            v.append(v_item)
        return m, v
    elif isinstance(model_params, np.ndarray):
        return np.zeros_like(model_params), np.zeros_like(model_params)
    else:
        return 0.0, 0.0
