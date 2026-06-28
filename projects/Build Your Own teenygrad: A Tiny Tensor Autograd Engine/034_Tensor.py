class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, LazyBuffer):
            self.lazydata = data
        else:
            self.lazydata = LazyBuffer(np.array(data, dtype=np.float32))
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
    
    @property
    def shape(self):
        return self.lazydata.shape
    
    @property
    def dtype(self):
        return self.lazydata.dtype
    
    def numpy(self):
        return self.lazydata._np
    
    @property
    def data(self):
        return self.lazydata
    
    @data.setter
    def data(self, value):
        if isinstance(value, LazyBuffer):
            self.lazydata = value
        else:
            self.lazydata = LazyBuffer(np.array(value, dtype=np.float32))
