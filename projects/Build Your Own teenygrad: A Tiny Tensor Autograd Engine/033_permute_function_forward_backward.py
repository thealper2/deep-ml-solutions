def permute_function_forward_backward():
    def forward(ctx, x, order):
        ctx.order = order
        return x.permute(order)

    def backward(ctx, grad_output):
        inverse_order = argsort(ctx.order)
        return grad_output.permute(inverse_order)

    return forward, backward
