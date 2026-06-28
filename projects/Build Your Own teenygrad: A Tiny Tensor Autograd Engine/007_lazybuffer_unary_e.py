def e(self, op):
    if op.name == 'NEG':
        return LazyBuffer(-self._np)
    elif op.name == 'RELU':
        return LazyBuffer(np.maximum(self._np, 0))
    elif op.name == 'LOG':
        return LazyBuffer(np.log(self._np))
    elif op.name == 'EXP':
        return LazyBuffer(np.exp(self._np))
    elif op.name == 'SQRT':
        return LazyBuffer(np.sqrt(self._np))
    elif op.name == 'SIGMOID':
        return LazyBuffer(1 / (1 + np.exp(-self._np)))
    else:
        raise ValueError(f'Unknown unary op: {op}')

LazyBuffer.e = e
