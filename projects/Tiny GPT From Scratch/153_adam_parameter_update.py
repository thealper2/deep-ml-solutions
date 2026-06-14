import numpy as np

def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    """Apply the Adam update: param - lr * m_hat / (sqrt(v_hat) + eps)."""
    new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_param
