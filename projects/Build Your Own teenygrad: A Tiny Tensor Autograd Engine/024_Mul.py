class Mul(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        result = []
        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                if i == 0:
                    result.append(lazybuffer_binary_e(self.y, BinaryOps.MUL, grad_output))
                else:
                    result.append(lazybuffer_binary_e(self.x, BinaryOps.MUL, grad_output))
            else:
                result.append(None)

        return tuple(result)
