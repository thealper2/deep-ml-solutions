import numpy as np
import torch


def predict_proba(model, X):
    """Probability of class 1 for each row, as a NumPy array, in eval mode without gradients."""
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1].numpy()