def tensor_randn(shape, seed=None, requires_grad=False):
    uniform = LazyBuffer.rand(shape, seed)

    u = uniform._np
    total_elements = np.prod(shape)
    if total_elements % 2 != 0:
        extra = LazyBuffer.rand((1,), seed).reshape
        pass

    rng = np.random.default_rng(seed)
    normal_arr = rng.normal(size=shape).astype(np.float32)
    return Tensor(LazyBuffer(normal_arr), requires_grad)
