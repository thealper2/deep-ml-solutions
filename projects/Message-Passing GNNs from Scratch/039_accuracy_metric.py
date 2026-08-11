def accuracy_metric(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().sum().item()
    total = targets.shape[0]
    return correct / total
