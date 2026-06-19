import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    probs = np.clip(probs, eps, 1.0 - eps)
    correct_probs = gather_true_class_probs(probs, labels)
    log_probs = np.log(correct_probs)
    loss = -np.mean(log_probs)
    if abs(loss) < eps:
        return -0.0

    return loss
