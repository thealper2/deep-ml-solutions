from tinygrad import Tensor

def random_batches(X, y, batch_size, n_batches):
    """Yield n_batches random mini-batches (xb, yb) from X and y."""
    N = X.shape[0]
    for _ in range(n_batches):
        idx = Tensor.randint(batch_size, high=N)
        yield X[idx], y[idx]