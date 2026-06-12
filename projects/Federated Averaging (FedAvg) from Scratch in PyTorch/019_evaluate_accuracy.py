def evaluate_accuracy(model, test_features, test_labels):
    logits = model(test_features)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == test_labels).float()
    accuracy = correct.mean().item()
    return accuracy
