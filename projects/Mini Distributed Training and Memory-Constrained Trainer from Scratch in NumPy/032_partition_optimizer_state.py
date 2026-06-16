def partition_optimizer_state(state, num_workers):
    m = state['m']
    v = state['v']
    t = state['t']

    workers = []
    for worker_idx in range(num_workers):
        worker_m = {}
        worker_v = {}
        worker_slices = {}
        worker_shapes = {}

        for key in m:
            arr_m = m[key]
            arr_v = v[key]
            total_elements = arr_m.size
            worker_shapes[key] = arr_m.shape

            base_size = total_elements // num_workers
            remainder = total_elements % num_workers

            start = worker_idx * base_size + min(worker_idx, remainder)
            end = start + base_size + (1 if worker_idx < remainder else 0)

            flat_m = arr_m.flatten()[start:end]
            flat_v = arr_v.flatten()[start:end]

            worker_m[key] = flat_m
            worker_v[key] = flat_v
            worker_slices[key] = (start, end)

        workers.append({
            'm': worker_m,
            'v': worker_v,
            't': t,
            'shard_slices': worker_slices,
            'shapes': worker_shapes,
        })

    return workers
