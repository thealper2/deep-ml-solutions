import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    def eval_poly(coeffs, x):
        val = 0.0
        for coeff in coeffs:
            val = val * x + coeff

        return val

    def eval_derivative(coeffs, x):
        n = len(coeffs)
        deriv_coeffs = [coeffs[i] * (n - 1 - i) for i in range(n - 1)]
        return eval_poly(deriv_coeffs, x)

    g = eval_poly(g_coeffs, x)
    h = eval_poly(h_coeffs, x)
    g_prime = eval_derivative(g_coeffs, x)
    h_prime = eval_derivative(h_coeffs, x)
    result = (g_prime * h - g * h_prime) / (h ** 2)
    return result
