import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """Small CNN: Conv2d -> ReLU -> pool -> (optional extras) -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc = nn.Linear(16 * 2 * 2, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def build_model(img_size=8, n_classes=2):
    """Return an instance of your TinyCNN (or equivalent nn.Module)."""
    return TinyCNN(img_size=img_size, n_classes=n_classes)


def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    """Train model on train_x/train_y and return the trained model."""
    torch.manual_seed(seed)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    N = train_x.shape[0]
    
    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(N)
        
        for i in range(0, N, batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = train_x[batch_indices]
            y_batch = train_y[batch_indices]
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    
    return model
