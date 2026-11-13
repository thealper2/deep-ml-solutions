import torch

def estimate_loss(model, splits, block_size, batch_size, eval_iters, generator=None):
    """Mean loss over eval_iters batches for each named split, in evaluation mode, as floats."""
    model.eval()
    out = {}
    with torch.no_grad():
        for name, data in splits.items():
            total = 0.0
            for _ in range(eval_iters):
                xb, yb = get_batch(data, block_size, batch_size, generator=generator)
                _, loss = model(xb, yb)
                total += float(loss.item())
            out[name] = total / eval_iters
    model.train()
    return out