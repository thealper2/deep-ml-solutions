def tensor_from_data(data, requires_grad=False):
    if isinstance(data, LazyBuffer):
        return Tensor(data, requires_grad)
    elif isinstance(data, Tensor):
        return Tensor(data.data, requires_grad)
    else:
        return Tensor(data, requires_grad)
