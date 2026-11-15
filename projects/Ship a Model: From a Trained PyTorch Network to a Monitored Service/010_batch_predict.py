import numpy as np
import torch


def batch_predict(model, rows, max_batch=32):
    """(probs in request order, number of forward passes), chunks of at most max_batch."""
    if len(rows) == 0:
        return [], 0

    model.eval()
    probs = []
    calls = 0
    with torch.no_grad():
        for i in range(0, len(rows), max_batch):
            chunk = rows[i:i + max_batch]
            xb = torch.as_tensor(np.asarray(chunk), dtype=torch.float32)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.tolist())
            calls += 1

    return probs, calls
