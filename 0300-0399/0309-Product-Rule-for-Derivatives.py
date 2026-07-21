import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    f = np.polynomial.Polynomial(f_coeffs)
    g = np.polynomial.Polynomial(g_coeffs)
    f_prime = f.deriv()
    g_prime = g.deriv()
    result = (f_prime * g) + (f * g_prime)
    return result.coef.tolist()
