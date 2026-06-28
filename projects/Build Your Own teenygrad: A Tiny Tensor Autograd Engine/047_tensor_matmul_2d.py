def tensor_matmul_2d(a, b):
    m, k = a.shape
    k2, n = b.shape

    a3 = a.reshape((m, k, 1)).expand((m, k, n))
    b3 = b.reshape((1, k, n)).expand((m, k, n))

    prod = Mul.apply(a3, b3)
    summed = Sum.apply(prod, axis=1)
    return summed.reshape((m, n))
