def bind_binary_tensor_methods():
    def make_binary_op(cls):
        def op(self, other):
            bx, by = broadcasted(self, other)
            return cls.apply(bx, by)

        return op

    Tensor.add = make_binary_op(Add)
    Tensor.sub = make_binary_op(Sub)
    Tensor.mul = make_binary_op(Mul)
    Tensor.div = make_binary_op(Div)

    Tensor.__add__ = Tensor.add
    Tensor.__sub__ = Tensor.sub
    Tensor.__mul__ = Tensor.mul
    Tensor.__truediv__ = Tensor.div
