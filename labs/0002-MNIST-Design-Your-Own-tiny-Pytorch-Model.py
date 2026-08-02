import torch
import torch.nn as nn
import torch.nn.functional as F

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.conv3 = nn.Conv2d(8, 16, 1)
            self.fc = nn.Linear(16, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.mean(dim=(2, 3))
            x = self.fc(x)
            return x
    
    return TinyNet()
