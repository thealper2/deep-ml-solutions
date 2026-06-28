def tensor_creation_helpers():
    def zeros_fn(shape):
        return Tensor(LazyBuffer.const(0.0, shape))

    def ones_fn(shape):
        return Tensor(LazyBuffer.const(1.0, shape))

    def full_fn(shape, value):
        return Tensor(LazyBuffer.const(value, shape))

    return zeros_fn, ones_fn, full_fn
