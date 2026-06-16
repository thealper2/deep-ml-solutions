def bucket_gradients(grads, bucket_size):
    sorted_keys = sorted(grads.keys())

    buckets = []
    meta = []

    current_bucket = []
    current_size = 0
    bucket_idx = 0
    start_pos = 0

    for key in sorted_keys:
        arr = grads[key].flatten()
        arr_len = arr.shape[0]

        if current_size + arr_len > bucket_size and current_bucket:
            buckets.append(np.concatenate(current_bucket))
            current_bucket = []
            current_size = 0
            start_pos = 0
            bucket_idx += 1

        current_bucket.append(arr)
        end_pos = start_pos + arr_len
        meta.append((key, grads[key].shape, start_pos, end_pos, bucket_idx))
        current_size += arr_len
        start_pos = end_pos

    if current_bucket:
        buckets.append(np.concatenate(current_bucket))

    return buckets, meta
