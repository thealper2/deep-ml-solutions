import torch
import torch.nn as nn

def init_conv_backbone(in_channels=2, hidden_channels=16):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )
