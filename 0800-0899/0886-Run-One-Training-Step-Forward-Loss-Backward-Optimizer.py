import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    return loss.item()