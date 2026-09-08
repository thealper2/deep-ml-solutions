"""
Fashion-MNIST Classifier in PyTorch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_fashion_mnist ──
import os
import gzip
import tempfile
import urllib.request
import numpy as np
import torch

def load_fashion_mnist(n_train=10000, n_test=2000):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    temp_dir = tempfile.gettempdir()
    data = {}

    for key, filename in files.items():
        filepath = os.path.join(temp_dir, filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)

        with gzip.open(filepath, "rb") as f:
            raw_data = f.read()
        
        if "images" in key:
            parsed = np.frombuffer(raw_data, dtype=np.uint8, offset=16)
            num_images = parsed.shape[0] // (28 * 28)
            parsed = parsed.reshape(num_images, 28, 28)
        else:
            parsed = np.frombuffer(raw_data, dtype=np.uint8, offset=8)

        data[key] = parsed

    return {
        "X_train": torch.tensor(data["train_images"][:n_train], dtype=torch.float32) / 255.0,
        "y_train": torch.tensor(data["train_labels"][:n_train], dtype=torch.int64),
        "X_test": torch.tensor(data["test_images"][:n_test], dtype=torch.float32) / 255.0,
        "y_test": torch.tensor(data["test_labels"][:n_test], dtype=torch.int64),
    }

# ── Step 002  FashionDataset ──
from torch.utils.data import Dataset

class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        y = self.y[i]
        return x, y

# ── Step 003  make_loaders ──
import torch
from torch.utils.data import DataLoader

def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    n_train = len(X_train)
    train_size = n_train - val_size

    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val_split = X_train[train_size:]
    y_val_split = y_train[train_size:]

    train_ds = FashionDataset(X_train_split, y_train_split)
    val_ds = FashionDataset(X_val_split, y_val_split)
    test_ds = FashionDataset(X_test, y_test)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "sizes": (len(train_ds), len(val_ds), len(test_ds))
    }

# ── Step 004  MLP ──
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Step 005  train_one_epoch ──
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

# ── Step 006  evaluate ──
import torch

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += len(yb)
    
    mean_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return mean_loss, accuracy

# ── Step 007  fit ──
import copy
import torch
import torch.nn as nn

def fit(model, loaders, epochs=5, lr=0.05, seed=42):
    torch.manual_seed(seed)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0
    }
    
    best_acc = -1.0
    best_state = None
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loaders['train'], loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, loaders['val'], loss_fn)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch
    
    # Restore best model
    model.load_state_dict(best_state)
    
    return history

# ── Step 008  lr_range_test ──
import torch
import torch.nn as nn

def lr_range_test(make_model, loader, lrs, n_batches=20, seed=42):
    results = {}
    
    for lr in lrs:
        torch.manual_seed(seed)
        model = make_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        total_loss = 0.0
        num_batches = 0
        
        for i, (xb, yb) in enumerate(loader):
            if i >= n_batches:
                break
            
            model.train()
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        results[lr] = total_loss / num_batches
    
    return results

# ── Step 009  random_search ──
import numpy as np
import torch

def random_search(loaders, n_trials=4, epochs=2, seed=42):
    rng = np.random.default_rng(seed)
    
    hidden1_options = [100, 200, 300]
    hidden2_options = [50, 100]
    lr_options = [0.01, 0.05, 0.1]
    
    trials = []
    best_trial = None
    best_acc = -1.0
    
    for _ in range(n_trials):
        hidden1 = rng.choice(hidden1_options)
        hidden2 = rng.choice(hidden2_options)
        lr = rng.choice(lr_options)
        
        torch.manual_seed(seed)
        model = MLP(hidden1, hidden2)
        
        history = fit(model, loaders, epochs=epochs, lr=lr, seed=seed)
        val_acc = max(history['val_acc'])
        
        trial = {
            'hidden1': int(hidden1),
            'hidden2': int(hidden2),
            'lr': float(lr),
            'val_acc': float(val_acc)
        }
        trials.append(trial)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_trial = trial
    
    return {
        'trials': trials,
        'best': best_trial
    }

# ── Step 010  test_accuracy ──
import torch.nn as nn

def test_accuracy(model, loaders):
    _, accuracy = evaluate(model, loaders['test'], nn.CrossEntropyLoss())
    return accuracy

# ── Step 011  save_model ──
import torch

def save_model(model, path):
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'hidden1': model.fc1.out_features,
            'hidden2': model.fc2.out_features,
            'n_classes': model.out.out_features
        }
    }, path)

def load_model(path):
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint['config']
    model = MLP(
        hidden1=config['hidden1'],
        hidden2=config['hidden2'],
        n_classes=config['n_classes']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# ── Step 012  predict_classes ──
import torch
import numpy as np

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def predict_classes(model, images):
    x = torch.tensor(images, dtype=torch.float32) / 255.0
    
    mean = 0.2860
    std = 0.3530
    x = (x - mean) / std
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
    
    return [CLASS_NAMES[idx] for idx in preds.tolist()]

# ── Scaffold (runner) ──
"""Fashion-MNIST classifier in PyTorch (Hands-On ML, chapter 10).

Story: the real Fashion-MNIST files, a Dataset with normalization and seeded
DataLoaders, an MLP nn.Module, one-epoch and evaluation loops, a fit function
that restores the best validation epoch, a learning-rate range test, a small
random search, one test-set score, and a saved model serving raw images.
"""
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn


def main() -> None:
    data = load_fashion_mnist(n_train=10000, n_test=2000)
    loaders = make_loaders(data, batch_size=64, val_size=2000)
    print(f"Fashion-MNIST slices: train/val/test = {loaders['sizes']}, {len(loaders['train'])} training batches per epoch")

    # ---- 1. Where should the learning rate be? ----
    curve = lr_range_test(MLP, loaders["train"], [0.001, 0.01, 0.05, 0.1, 0.5, 2.0], n_batches=20)
    print("LR range test (mean loss over 20 batches): " + "  ".join(f"{lr}:{v:.2f}" for lr, v in curve.items()))

    # ---- 2. Train with validation and best-epoch restore ----
    torch.manual_seed(0)
    model = MLP()
    print(f"\nMLP 784-300-100-10 with {count_parameters(model):,} parameters")
    hist = fit(model, loaders, epochs=5, lr=0.05)
    for e, (tl, vl, va) in enumerate(zip(hist["train_loss"], hist["val_loss"], hist["val_acc"])):
        print(f"  epoch {e + 1}: train loss {tl:.3f}  val loss {vl:.3f}  val acc {va:.3f}")
    print(f"restored weights from epoch {hist['best_epoch'] + 1}")

    # ---- 3. A small random search ----
    search = random_search(loaders, n_trials=3, epochs=2)
    for t in search["trials"]:
        print(f"  trial hidden={t['hidden1']}/{t['hidden2']} lr={t['lr']}: val acc {t['val_acc']:.3f}")
    print(f"best config: {search['best']}")

    # ---- 4. Test once, ship ----
    print(f"\nTEST accuracy {test_accuracy(model, loaders):.3f} (best validation was {max(hist['val_acc']):.3f})")
    path = os.path.join(tempfile.gettempdir(), "fashion_mlp.pt")
    save_model(model, path)
    served = load_model(path)
    raw = (data["X_test"][:6] * 255).round().to(torch.uint8).numpy()
    preds = predict_classes(served, raw)
    truth = [CLASS_NAMES[i] for i in data["y_test"][:6].tolist()]
    for p, t in zip(preds, truth):
        print(f"  predicted {p:<12} truth {t}")


if __name__ == "__main__":
    main()
