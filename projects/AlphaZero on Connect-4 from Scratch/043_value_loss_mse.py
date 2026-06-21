import torch

def value_loss_mse(predicted_values, target_values):
    return torch.mean((predicted_values - target_values) ** 2)
