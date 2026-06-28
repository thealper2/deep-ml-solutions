def r(self, op, axis):
    if op.name == 'SUM':
        result = np.sum(self._np, axis=axis, keepdims=True)
    elif op.name == 'MAX':
        result = np.max(self._np, axis=axis, keepdims=True)
    else:
        raise ValueError(f'Unknown reduce op: {op}')
    
    return LazyBuffer(result)
