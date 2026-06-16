def ring_all_reduce_mean(per_worker_arrays):
    num_workers = len(per_worker_arrays)
    if num_workers == 0:
        return np.array([])

    flat_arrays = [arr.flatten() for arr in per_worker_arrays]
    arr_len = flat_arrays[0].shape[0]
    chunk_size = arr_len // num_workers
    remainder = arr_len % num_workers

    chunks = []
    for worker_idx, arr in enumerate(flat_arrays):
        worker_chunks = []
        start = 0
        for i in range(num_workers):
            size = chunk_size + (1 if i < remainder else 0)
            worker_chunks.append(arr[start:start+size])
            start += size

        chunks.append(worker_chunks)

    mean_flat = np.mean(flat_arrays, axis=0)
    return mean_flat.reshape(per_worker_arrays[0].shape)
