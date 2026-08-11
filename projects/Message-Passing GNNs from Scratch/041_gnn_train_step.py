def gnn_train_step(params, batch, forward_fn, loss_fn, lr):
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    predictions = forward_fn(params, batch)
    loss = loss_fn(predictions, batch['y'])
    loss.backward()

    with torch.no_grad():
        for p in params.values():
            if p.grad is not None:
                p.sub_(lr * p.grad)

    return {'loss': float(loss.item()), 'params': params}
