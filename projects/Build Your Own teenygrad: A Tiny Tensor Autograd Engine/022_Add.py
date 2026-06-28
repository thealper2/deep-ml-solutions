class Add(Function):
    def forward(self, x, y):
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output):
        result = []
        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                result.append(grad_output)
            else:
                result.append(None)

        return tuple(result)
