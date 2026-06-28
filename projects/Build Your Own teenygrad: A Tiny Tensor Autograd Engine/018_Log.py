class Log(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.x = x
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        inv_x = self.x.e(UnaryOps.NEG)
        one = LazyBuffer.const(1.0, self.x.shape)
        inv_x = lazybuffer_binary_e(one, BinaryOps.DIV, self.x)
        return lazybuffer_binary_e(inv_x, BinaryOps.MUL, grad_output)
