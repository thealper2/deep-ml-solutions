def iterate_minibatches(x, y, batch_size, seed=0):
    N = x.shape[0]
    idx = shuffle_indices(N, seed)

    for start in range(0, N - batch_size + 1, batch_size):
        if start + batch_size > N:
            break
        end = start + batch_size
        batch_idx = idx[start:end]
        yield x[batch_idx], y[batch_idx]
