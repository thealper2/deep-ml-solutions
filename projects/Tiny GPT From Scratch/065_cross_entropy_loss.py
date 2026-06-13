import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    correct_probs = gather_correct_token_probs(probs, targets)
    log_probs = array_log(correct_probs)
    return -np.mean(log_probs)
