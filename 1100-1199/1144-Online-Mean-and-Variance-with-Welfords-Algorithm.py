def welford_stats(data, ddof=1):
    n = 0
    mean = 0.0
    m2 = 0.0

    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2

    if n == 0:
        return (0, 0.0, 0.0)

    divisor = n - ddof
    if divisor <= 0:
        variance = 0.0
    else:
        variance = m2 / divisor

    return (n, mean, variance)