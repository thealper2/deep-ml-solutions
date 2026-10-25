import numpy as np


def cross_entropy(logits, targets):
    B, T, V = logits.shape
    return -(logits.logsoftmax(-1) * targets).sum() * (1.0 / (B * T))


def build_train_program(B, T, V, d, lr):
    import numpy as np

    x = Tensor.input('x', (B, T, V))
    y = Tensor.input('y', (B, T, V))
    mask = Tensor.input('mask', (T, T))

    shapes = gpt_init(V, T, d, np.random.default_rng(0))
    params = {k: Tensor.input(k, v.shape) for k, v in shapes.items()}

    logits = gpt_forward(x, mask, params)
    loss = cross_entropy(logits, y)

    grads = backward(loss, list(params.values()))

    outputs = {'loss': loss}
    for name, p in params.items():
        g = grads[p]
        outputs[name + '_new'] = p - lr * g

    return schedule(outputs)