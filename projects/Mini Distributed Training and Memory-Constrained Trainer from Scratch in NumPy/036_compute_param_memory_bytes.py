def compute_param_memory_bytes(params):
    total_bytes = 0
    for arr in params.values():
        total_bytes += arr.nbytes

    return total_bytes
