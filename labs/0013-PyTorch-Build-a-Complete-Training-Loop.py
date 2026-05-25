import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_train = X_train.shape[0]

    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        indices = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            batch_indices = indices[:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_indices)

        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        correct = 0
        n_val = X_val.shape[0]

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                X_batch = X_val[i:i+batch_size]
                y_batch = y_val[i:i+batch_size]

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(batch_indices)

                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()

        val_loss /= n_val
        val_accuracy = correct / n_val

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })

    return history

