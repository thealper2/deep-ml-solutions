def expand_function_forward(ctx, x, shape):
    ctx.input_shape = x.shape
    return x.expand(shape)
