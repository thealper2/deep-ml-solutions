def subtract_params(params_a, params_b):
    return {key: params_a[key] - params_b[key] for key in params_a}
