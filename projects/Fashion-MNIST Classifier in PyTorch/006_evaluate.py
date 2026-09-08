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