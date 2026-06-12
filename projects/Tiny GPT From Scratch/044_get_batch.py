def get_batch(data, block_size, batch_size, rng):
    data_len = len(data)
    offsets = sample_random_batch_offsets(data_len, block_size, batch_size, rng)
    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)
    return X, Y
