class Exp(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.EXP)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)
