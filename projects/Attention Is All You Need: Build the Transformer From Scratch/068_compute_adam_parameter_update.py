import torch

def compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon):
    """Return delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon); the caller subtracts it."""
    with torch.no_grad():
        return learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
