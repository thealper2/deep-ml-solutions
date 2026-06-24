import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    loss = torch.sum(log_probabilities * smoothed_distribution)
    return -loss if loss != 0 else loss
