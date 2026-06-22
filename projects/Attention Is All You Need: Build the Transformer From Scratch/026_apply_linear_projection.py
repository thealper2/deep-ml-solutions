def apply_linear_projection(x, weight, bias):
    z = x @ weight.T
    if bias is not None:
        z = z + bias

    return z
