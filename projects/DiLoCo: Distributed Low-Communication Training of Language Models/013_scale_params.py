def scale_params(params, scalar):
    return {key: value * scalar for key, value in params.items()}
