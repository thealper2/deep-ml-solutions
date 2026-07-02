def accuracy(predictions, labels):
    return (predictions == labels).sum() / len(labels)
