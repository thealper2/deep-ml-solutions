def split_into_micro_batches(x, y, micro_batch_size):
    N = x.shape[0]
    batches = []
    for i in range(0, N, micro_batch_size):
        x_batch = x[i:i+micro_batch_size]
        y_batch = y[i:i+micro_batch_size]
        batches.append((x_batch, y_batch))

    return batches
