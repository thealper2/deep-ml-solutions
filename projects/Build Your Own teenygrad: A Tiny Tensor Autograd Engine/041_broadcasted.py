def broadcasted(x, y):
    x_shape = x.shape
    y_shape = y.shape
    
    if x_shape == y_shape:
        return x, y
    
    max_ndim = max(len(x_shape), len(y_shape))
    x_padded = (1,) * (max_ndim - len(x_shape)) + x_shape
    y_padded = (1,) * (max_ndim - len(y_shape)) + y_shape
    
    out_shape = []
    for i in range(max_ndim):
        if x_padded[i] == 1:
            out_shape.append(y_padded[i])
        elif y_padded[i] == 1:
            out_shape.append(x_padded[i])
        elif x_padded[i] == y_padded[i]:
            out_shape.append(x_padded[i])
        else:
            raise ValueError(f"Cannot broadcast shapes {x_shape} and {y_shape}")
    
    out_shape = tuple(out_shape)
    
    if x_shape == out_shape:
        bx = x
    else:
        if len(x_shape) < len(out_shape):
            new_shape = (1,) * (len(out_shape) - len(x_shape)) + x_shape
            x_reshaped = x.reshape(new_shape)
            bx = x_reshaped.expand(out_shape)
        else:
            bx = x.expand(out_shape)
    
    if y_shape == out_shape:
        by = y
    else:
        if len(y_shape) < len(out_shape):
            new_shape = (1,) * (len(out_shape) - len(y_shape)) + y_shape
            y_reshaped = y.reshape(new_shape)
            by = y_reshaped.expand(out_shape)
        else:
            by = y.expand(out_shape)
    
    return bx, by
