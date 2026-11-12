import torch
import torch.nn.functional as F

def evaluate(model, X, Y):
    """Mean cross-entropy in evaluation mode, as a float; layers back in training mode afterwards."""
    for l in model.layers:
        if hasattr(l, 'training'):
            l.training = False
    with torch.no_grad():
        logits = model(X)
        loss = float(F.cross_entropy(logits, Y).item())
    for l in model.layers:
        if hasattr(l, 'training'):
            l.training = True
    return loss
