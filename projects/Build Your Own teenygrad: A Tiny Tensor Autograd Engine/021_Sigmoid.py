class Sigmoid(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        one = LazyBuffer.const(1.0, self.ret.shape)
        one_minus_ret = lazybuffer_binary_e(one, BinaryOps.SUB, self.ret)
        grad = lazybuffer_binary_e(self.ret, BinaryOps.MUL, one_minus_ret)
        return lazybuffer_binary_e(grad, BinaryOps.MUL, grad_output)
