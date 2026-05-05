import math

def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    if x < 0:
        return 0.0

    exponent = -x / 2
    term1 = x ** ((k / 2) - 1)
    term2 = math.exp(exponent)
    denominator = (2 **  (k / 2)) * math.gamma(k / 2)
    pdf = (1 / denominator) * term1 * term2
    return round(pdf, 3)