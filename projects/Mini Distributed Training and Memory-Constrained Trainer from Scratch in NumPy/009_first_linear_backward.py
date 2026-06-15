def first_linear_backward(d_z1, x, w1):
    dx = d_z1 @ w1.T
    dW1 = x.T @ d_z1
    db1 = np.sum(d_z1, axis=0)
    return dx, dW1, db1
