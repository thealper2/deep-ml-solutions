def rand(shape, seed=None):
    rng = np.random.default_rng(seed)
    return LazyBuffer(rng.random(size=shape, dtype=np.float32))
