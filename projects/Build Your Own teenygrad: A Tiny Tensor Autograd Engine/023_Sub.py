class Sub(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        UnaryOps, _, _, _ = make_op_enums()
        result = []
        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                if i == 0:
                    result.append(grad_output)
                else:
                    result.append(grad_output.e(UnaryOps.NEG))

            else:
                result.append(None)

        return tuple(result)
