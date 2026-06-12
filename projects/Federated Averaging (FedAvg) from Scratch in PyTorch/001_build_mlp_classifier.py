import torch
import torch.nn as nn


def build_mlp_classifier(input_size, hidden_size, num_classes):
    class _MLPClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    return _MLPClassifier(input_size, hidden_size, num_classes)
