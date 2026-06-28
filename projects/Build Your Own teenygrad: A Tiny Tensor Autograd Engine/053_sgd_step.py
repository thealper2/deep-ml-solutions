def sgd_step(parameters, learning_rate):
    for p in parameters:
        if p.grad is None:
            continue
        
        updated = p.data._np - learning_rate * p.grad.numpy()
        p.data = LazyBuffer(updated)


    return None
