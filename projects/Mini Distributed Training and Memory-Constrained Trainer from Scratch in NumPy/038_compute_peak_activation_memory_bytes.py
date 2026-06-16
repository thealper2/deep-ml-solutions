def compute_peak_activation_memory_bytes(x, params, checkpointed=False):
    if not checkpointed:
        _, cache = mlp_forward(x, params)
        total_bytes = 0
        for arr in cache.values():
            if isinstance(arr, np.ndarray):
                total_bytes += arr.nbytes

        return total_bytes
    else:
        light_cache = {'x': x}
        total_bytes = x.nbytes
        return total_bytes
