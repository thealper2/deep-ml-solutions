def permute(self, order):
    return LazyBuffer(np.transpose(self._np, order))
