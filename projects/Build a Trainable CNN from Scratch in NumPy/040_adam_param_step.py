import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)
