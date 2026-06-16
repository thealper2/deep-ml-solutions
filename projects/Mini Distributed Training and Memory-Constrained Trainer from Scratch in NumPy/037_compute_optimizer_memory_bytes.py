def compute_optimizer_memory_bytes(state, num_workers=1, sharded=False):
    total_bytes = 0
    for key in ['m', 'v']:
        for arr in state[key].values():
            total_bytes += arr.nbytes

    if sharded:
        total_bytes = total_bytes // num_workers

    return total_bytes
