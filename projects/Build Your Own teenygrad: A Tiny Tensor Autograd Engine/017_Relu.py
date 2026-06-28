class Relu(Function):
    def forward(self, x):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        zero = LazyBuffer.const(0.0, self.ret.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)
