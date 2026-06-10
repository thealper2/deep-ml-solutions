from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam
import numpy as np

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a tinygrad model and return per-epoch metrics.

    Steps:
    1. Create optimizer: Adam(get_parameters(model), lr=lr).
    2. For each epoch:
       a. Shuffle training data (Tensor.randperm or numpy shuffle).
       b. Iterate mini-batches: zero_grad -> forward -> loss -> backward -> step.
       c. Set Tensor.training = False, evaluate on val set.
       d. Append {'epoch', 'train_loss', 'val_loss', 'val_accuracy'} to history.
    3. Return the history list.
    """
    optimizer = Adam(get_parameters(model), lr=lr)
    history = []

    n_train = X_train.shape[0]

    for epoch in range(epochs):
        Tensor.training = True

        perm = np.random.permutation(n_train).tolist()

        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        train_loss = 0.0
        batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)

            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]

            optimizer.zero_grad()

            logits = model(xb)
            loss = logits.sparse_categorical_crossentropy(yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        train_loss /= batches

        Tensor.training = False

        val_logits = model(X_val)
        val_loss = val_logits.sparse_categorical_crossentropy(y_val).item()

        preds = val_logits.argmax(axis=1)
        val_acc = (preds == y_val).mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

    return history
