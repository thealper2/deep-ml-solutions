import torch.nn as nn

def test_accuracy(model, loaders):
    _, accuracy = evaluate(model, loaders['test'], nn.CrossEntropyLoss())
    return accuracy