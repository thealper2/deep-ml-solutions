def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    notes = (
        "For Y = X @ W, where X is (B, D_in), W is (D_in, D_out), Y is (B, D_out).\n"
        "By the chain rule, the gradient dL/dW is obtained by backpropagating dL/dY through the linear transformation:\n"
        "dL/dW = X.T @ dY.\n"
        "This can be verified elementwise: Y_{i,k} = sum_j X_{i,j} W_{j,k}, so dL/dW_{j,k} = sum_i X_{i,j} * dL/dY_{i,k}."
    )
    return notes
