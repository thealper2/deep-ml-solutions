def backward(loss, params):
    # seed gradient of loss with constant 1.0
    grads = {loss.uop: Tensor.const(1.0, loss.dtype)}

    order = loss.uop.toposort()

    for u in reversed(order):
        if u not in grads:
            continue
        g = grads[u]

        if u.op in ALU:
            pairs = grad_alu(u, g)
        elif u.op in MOVEMENT or u.op is Ops.REDUCE_AXIS:
            pairs = grad_movement(u, g)
        else:
            pairs = []

        for src, gt in pairs:
            gt = gt.expand(*shape_of(src))
            if src in grads:
                grads[src] = grads[src] + gt
            else:
                grads[src] = gt

    result = {}
    for p in params:
        result[p] = grads.get(p.uop)
    return result