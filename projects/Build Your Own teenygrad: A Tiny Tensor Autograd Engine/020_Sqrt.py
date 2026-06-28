class Sqrt(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        two = LazyBuffer.const(2.0, self.ret.shape)
        denom = lazybuffer_binary_e(two, BinaryOps.MUL, self.ret)
        one = LazyBuffer.const(1.0, self.ret.shape)
        inv_denom = lazybuffer_binary_e(one, BinaryOps.DIV, denom)
        return lazybuffer_binary_e(inv_denom, BinaryOps.MUL, grad_output)
