def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    w = probe_params['w'].clone().detach().requires_grad_(True)
    b = probe_params['b'].clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        pred = embeddings @ w + b
        loss = torch.mean((pred - states) ** 2)
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad = None
            b.grad = None

    return {'w': w.detach(), 'b': b.detach()}
