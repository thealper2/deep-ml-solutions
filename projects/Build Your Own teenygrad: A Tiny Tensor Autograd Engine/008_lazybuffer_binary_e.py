def lazybuffer_binary_e(self, op, other):
    if op.name == 'ADD':
        return LazyBuffer(self._np + other._np)
    elif op.name == 'SUB':
        return LazyBuffer(self._np - other._np)
    elif op.name == 'MUL':
        return LazyBuffer(self._np * other._np)
    elif op.name == 'DIV':
        return LazyBuffer(self._np / other._np)
    elif op.name == 'CMPLT':
        return LazyBuffer((self._np < other._np).astype(np.float32))
    elif op.name == 'MAX':
        return LazyBuffer(np.maximum(self._np, other._np))
    else:
        raise ValueError(f'Unknown binary op: {op}')
