import torch.nn as nn

def compute_batch_loss(model, batch_features, batch_labels):
    criterion = nn.CrossEntropyLoss()
    logits = model(batch_features)
    loss = criterion(logits, batch_labels)
    return loss
