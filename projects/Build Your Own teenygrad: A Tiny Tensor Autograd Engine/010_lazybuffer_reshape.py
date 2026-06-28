def reshape(self, new_shape):
    if -1 in new_shape:
        total = prod(self.shape)
        known = prod([d for d in new_shape if d != -1])
        inferred = total // known
        new_shape = tuple(inferred if d == -1 else d for d in new_shape)

    return LazyBuffer(self._np.reshape(new_shape))
