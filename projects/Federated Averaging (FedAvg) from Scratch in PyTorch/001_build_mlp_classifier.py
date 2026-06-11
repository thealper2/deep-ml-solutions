import torch
import torch.nn as nn


def build_mlp_classifier(input_size, hidden_size, num_classes):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size), 
        nn.ReLU(), 
        nn.Linear(hidden_size, num_classes)
    )
