import torch
import torch.nn as nn


def detach_state(state):
    if state is None:
        return None
    h, c = state
    return (h.detach(), c.detach())


def chunk_stream(x, y, k):
    T = x.shape[1]
    chunks = []
    for start in range(0, T - k + 1, k):
        chunks.append((x[:, start:start + k, :], y[:, start:start + k, :]))
    return chunks


def train_tbptt(lstm, head, x, y, k, lr, epochs=1):
    params = list(lstm.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    chunks = chunk_stream(x, y, k)

    losses = []
    for _ in range(epochs):
        state = None
        for x_chunk, y_chunk in chunks:
            out, state = lstm(x_chunk, state)
            preds = head(out)
            loss = torch.nn.functional.mse_loss(preds, y_chunk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = detach_state(state)
            losses.append(float(loss.item()))
    return losses


def grad_reaches_previous_chunk(lstm, head, x, y, k):
    x1 = x[:, :k, :].clone().detach().requires_grad_(True)
    y1 = y[:, :k, :]
    x2 = x[:, k:2 * k, :]
    y2 = y[:, k:2 * k, :]

    out1, state = lstm(x1, None)
    state = detach_state(state)
    out2, _ = lstm(x2, state)
    preds = head(out2)
    loss = torch.nn.functional.mse_loss(preds, y2)
    loss.backward()

    if x1.grad is None:
        return False
    return bool(x1.grad.abs().sum().item() > 0)