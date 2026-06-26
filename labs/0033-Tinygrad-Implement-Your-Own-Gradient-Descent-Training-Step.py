from tinygrad import Tensor
from tinygrad.nn.state import get_parameters

def train_step(model, x_batch, y_batch, lr):
    """
    Perform ONE step of gradient descent training.

    Steps:
    1. Zero gradients (set p.grad = None for each parameter).
    2. Forward pass: logits = model(x_batch).
    3. Compute loss: logits.sparse_categorical_crossentropy(y_batch).
    4. Backward pass: loss.backward().
    5. Update each parameter manually using p.assign(p.detach() - lr * p.grad).
    6. Realize the updates (Tensor.realize(*params)) and return loss.item().

    Gotcha: tinygrad's nn.Linear / Conv2d / BatchNorm2d create parameters with
    requires_grad=None. The Optimizer base class flips this to True when you
    pass params in, but since you are NOT using nn.optim here you have to do
    it yourself - otherwise backward() will leave .grad as None and your update
    loop will be a no-op.
    """
    params = get_parameters(model)
    for p in params:
        p.requires_grad = True

    for p in params:
        p.grad = None

    logits = model(x_batch)
    loss = logits.sparse_categorical_crossentropy(y_batch)
    loss.backward()

    for p in params:
        if p.grad is not None:
            new_p = p.detach() - lr * p.grad
            p.assign(new_p)

    Tensor.realize(*params)
    return loss.item()
