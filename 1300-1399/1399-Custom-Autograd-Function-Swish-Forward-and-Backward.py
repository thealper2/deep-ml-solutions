import torch


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        local_grad = s * (1 + x * (1 - s))
        return grad_output * local_grad


def swish(x):
    return SwishFunction.apply(x)