import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    return torch.mean((noise - noise_pred) ** 2)
