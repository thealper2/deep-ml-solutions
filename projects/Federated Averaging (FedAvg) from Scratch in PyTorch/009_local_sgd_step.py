import torch.nn as nn
import torch.optim as optim

def local_sgd_step(model, optimizer, batch_features, batch_labels):
    optimizer.zero_grad()
    loss = compute_batch_loss(model, batch_features, batch_labels)
    loss.backward()
    optimizer.step()
    return loss.item()
