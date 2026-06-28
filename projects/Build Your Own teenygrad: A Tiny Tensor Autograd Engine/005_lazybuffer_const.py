def const(value, shape):
    return LazyBuffer(np.full(shape, value, dtype=np.float32))

LazyBuffer.const = staticmethod(const)
