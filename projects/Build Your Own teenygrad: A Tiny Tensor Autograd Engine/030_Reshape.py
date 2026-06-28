class Reshape(Function):
    def forward(self, x, shape):
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
