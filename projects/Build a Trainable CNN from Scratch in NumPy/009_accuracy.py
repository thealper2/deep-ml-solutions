def accuracy(logits_or_probs, labels):
    y_pred = argmax_rows(logits_or_probs)
    correct = (y_pred == labels).sum()
    return correct / len(labels)
