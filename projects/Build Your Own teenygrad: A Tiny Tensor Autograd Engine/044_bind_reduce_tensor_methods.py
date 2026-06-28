def bind_reduce_tensor_methods():
    def normalize_axis(axis, ndim):
        if axis is None:
            return None
        if isinstance(axis, int):
            if axis < 0:
                axis = axis + ndim
            if axis < 0 or axis >= ndim:
                raise ValueError(f"Axis {axis} out of bounds for ndim {ndim}")
            return axis
        if isinstance(axis, (tuple, list)):
            normalized = []
            for a in axis:
                if a < 0:
                    a = a + ndim
                if a < 0 or a >= ndim:
                    raise ValueError(f"Axis {a} out of bounds for ndim {ndim}")
                normalized.append(a)
            return tuple(normalized)
        raise ValueError(f"Invalid axis: {axis}")

    def drop_axes(result, norm_axis, ndim):
        if norm_axis is None:
            axes = list(range(ndim))
        elif isinstance(norm_axis, int):
            axes = [norm_axis]
        else:
            axes = list(norm_axis)
        new_shape = list(result.shape)
        for ax in sorted(axes, reverse=True):
            del new_shape[ax]
        return result.reshape(tuple(new_shape))

    def sum_method(self, axis=None, keepdim=False):
        ndim = len(self.shape)
        norm_axis = normalize_axis(axis, ndim)
        result = Sum.apply(self, axis=norm_axis)
        if not keepdim:
            result = drop_axes(result, norm_axis, ndim)
        return result

    def max_method(self, axis=None, keepdim=False):
        ndim = len(self.shape)
        norm_axis = normalize_axis(axis, ndim)
        result = Max.apply(self, axis=norm_axis)
        if not keepdim:
            result = drop_axes(result, norm_axis, ndim)
        return result

    Tensor.sum = sum_method
    Tensor.max = max_method
