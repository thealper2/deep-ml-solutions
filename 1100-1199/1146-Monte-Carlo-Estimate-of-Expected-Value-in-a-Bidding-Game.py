def bidding_mc(bid, value, low, high, n, antithetic=False):
    def van_der_corput(i):
        u = 0.0
        bit = 0.5
        while i > 0:
            if i & 1:
                u += bit

            i >>= 1
            bit *= 0.5

        return u

    payoffs = []
    for i in range(1, n + 1):
        u = van_der_corput(i)
        if antithetic:
            V1 = low + u * (high - low)
            V2 = low + (1 - u) * (high - low)
            payoff1 = value - bid if bid >= V1 else 0.0
            payoff2 = value - bid if bid >= V2 else 0.0
            paired_payoff = (payoff1 + payoff2) / 2.0
            payoffs.append(paired_payoff)
        else:
            V = low + u * (high - low)
            payoff = value - bid if bid >= V else 0.0
            payoffs.append(payoff)

    m = len(payoffs)

    if m == 0:
        return [0.0, 0.0]

    estimate = sum(payoffs) / m

    if m < 2:
        standard_error = 0.0
    else:
        variance = sum((p - estimate) ** 2 for p in payoffs) / (m - 1)
        standard_error = (variance ** 0.5) / (m ** 0.5)

    return [estimate, standard_error]