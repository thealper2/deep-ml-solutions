class Sum(Function):
    def forward(self, x, axis):
        self.input_shape = x.shape
        self.axis = axis
        result = np.sum(x._np, axis=axis, keepdims=True)
        return LazyBuffer(result)
