def expand(self, new_shape):
    if len(self.shape) != len(new_shape):
        raise ValueError(f"Cannot expand from {self.shape} to {new_shape}")

    for old, new in zip(self.shape, new_shape):
        if old != new and old != 1:
            raise ValueError(f"Cannot expand axis of size {old} to {new}")

    expanded = np.broadcast_to(self._np, new_shape)
    return LazyBuffer(expanded)
