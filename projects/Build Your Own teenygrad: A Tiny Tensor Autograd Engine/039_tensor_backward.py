def tensor_backward(tensor):
    ones_data = np.ones_like(tensor.numpy())
    tensor.grad = Tensor(ones_data, requires_grad=False)

    order = build_topological_order(tensor)

    for node in reversed(order):
        if node._ctx is not None and node.grad is not None:
            grad_lazy = LazyBuffer(node.grad.numpy())

            grad_outputs = node._ctx.backward(grad_lazy)

            if not isinstance(grad_outputs, tuple):
                grad_outputs = (grad_outputs,)

            for parent, grad_out in zip(node._ctx.parents, grad_outputs):
                if grad_out is not None and parent.requires_grad:
                    if isinstance(grad_out, LazyBuffer):
                        grad_out_np = grad_out._np
                    elif hasattr(grad_out, 'numpy'):
                        grad_out_np = grad_out.numpy()
                    else:
                        grad_out_np = np.array(grad_out)

                    if parent.grad is None:
                        parent.grad = Tensor(grad_out_np, requires_grad=False)
                    else:
                        parent.grad = Tensor(parent.grad.numpy() + grad_out_np, requires_grad=False)
