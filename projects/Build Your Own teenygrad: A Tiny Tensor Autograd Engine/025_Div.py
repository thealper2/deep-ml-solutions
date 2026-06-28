class Div(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        result = []

        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                if i == 0:
                    one = LazyBuffer.const(1.0, self.y.shape)
                    inv_y = lazybuffer_binary_e(one, BinaryOps.DIV, self.y)
                    result.append(lazybuffer_binary_e(inv_y, BinaryOps.MUL, grad_output))
                else:
                    div_result = lazybuffer_binary_e(self.x, BinaryOps.DIV, self.y)
                    neg_div = div_result.e(UnaryOps.NEG)
                    result.append(lazybuffer_binary_e(neg_div, BinaryOps.DIV, self.y))
            else:
                result.append(None)

        return tuple(result)
