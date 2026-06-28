def expand_function_backward(ctx, grad_output):
    input_shape = ctx.input_shape
    grad_shape = grad_output.shape

    axes_to_reduce = []
    for i, (in_dim, out_dim) in enumerate(zip(input_shape, grad_shape)):
        if in_dim == 1 and out_dim > 1:
            axes_to_reduce.append(i)

    reduced = grad_output
    for axis in sorted(axes_to_reduce, reverse=True):
        reduced = reduced.r(ReduceOps.SUM, axis)

    return reduced.reshape(input_shape)
