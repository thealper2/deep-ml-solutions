def accumulate_gradients(accum_grads, new_grads):
    if accum_grads is None:
        return {k: v.copy() for k, v in new_grads.items()}

    result = {}
    for key in accum_grads.keys():
        result[key] = accum_grads[key] + new_grads[key]

    return result
