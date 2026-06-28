def bind_movement_tensor_methods():
    class Expand(Function):
        def forward(self, x, shape):
            self.input_shape = x.shape
            return x.expand(shape)

        def backward(self, grad_output):
            input_shape = self.input_shape
            grad_shape = grad_output.shape

            axes_to_reduce = []
            for i, (in_dim, out_dim) in enumerate(zip(input_shape, grad_shape)):
                if in_dim == 1 and out_dim > 1:
                    axes_to_reduce.append(i)

            reduced = grad_output
            for axis in sorted(axes_to_reduce, reverse=True):
                reduced = reduced.r(ReduceOps.SUM, axis)

            return reduced.reshape(input_shape)

    class Permute(Function):
        def forward(self, x, order):
            self.order = order
            return x.permute(order)

        def backward(self, grad_output):
            inverse_order = argsort(self.order)
            return grad_output.permute(inverse_order)

    def reshape_method(self, *args):
        shape = args[0] if len(args) == 1 else args
        return Reshape.apply(self, shape=shape)

    def expand_method(self, *args):
        shape = args[0] if len(args) == 1 else args
        return Expand.apply(self, shape=shape)

    def permute_method(self, *args):
        order = args[0] if len(args) == 1 else args
        return Permute.apply(self, order=order)

    return {
        'reshape': reshape_method,
        'expand': expand_method,
        'permute': permute_method,
    }
