def init_outer_optimizer(params):
    momentum = {}
    for key in params:
        momentum[key] = np.zeros_like(params[key])

    return {'momentum': momentum}
