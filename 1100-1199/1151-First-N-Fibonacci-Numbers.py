def first_n_fibonacci(n):
    if n == 0:
        return []

    if n == 1:
        return [0]

    if n == 2:
        return [0, 1]

    f = [0, 1] + [0] * (n - 2)

    for i in range(2, n):
        f[i] = f[i - 1] + f[i - 2]

    return f
