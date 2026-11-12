import torch
import torch.nn.functional as F

def evaluate_loss(model, X, Y):
    """Mean cross-entropy of model(X) against Y, as a float, without gradients."""
    with torch.no_grad():
        logits = model(X)
        return float(F.cross_entropy(logits, Y).item())