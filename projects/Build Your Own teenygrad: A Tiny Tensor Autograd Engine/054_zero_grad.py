def zero_grad(parameters):
    for p in parameters:
        p.grad = None

    return None
