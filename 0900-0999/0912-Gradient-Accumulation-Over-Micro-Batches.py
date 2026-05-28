import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    model.zero_grad()
    num_batches = len(micro_batches)
    total_loss = 0.0

    for x, y in micro_batches:
        output = model(x)
        loss = criterion(output, y)
        loss_scaled = loss / num_batches
        loss_scaled.backward()
        total_loss += loss.item()

    optimizer.step()
    return total_loss / num_batches
