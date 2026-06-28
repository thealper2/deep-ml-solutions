def bind_unary_tensor_methods():
    def make_unary_op(cls):
        def op(tensor):
            return cls.apply(tensor)

        return op

    return {
        'neg': make_unary_op(Neg),
        'relu': make_unary_op(Relu),
        'log': make_unary_op(Log),
        'exp': make_unary_op(Exp),
        'sqrt': make_unary_op(Sqrt),
        'sigmoid': make_unary_op(Sigmoid),
    }
