import math

def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    """
    Compute Beta distribution statistics.
    
    Args:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (alpha > 0)
        beta_param: Second shape parameter (beta > 0)
    
    Returns:
        Dictionary with 'pdf', 'mean', and 'variance'
    """
    if x <= 0 or x >= 1:
        pdf = 0.0
    else:
        log_beta = math.lgamma(alpha) + math.lgamma(beta_param) - math.lgamma(alpha + beta_param)
        log_pdf = (alpha - 1) * math.log(x) + (beta_param - 1) * math.log(1 - x) - log_beta
        pdf = math.exp(log_pdf)

    mean = alpha / (alpha + beta_param)
    variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))

    return {
        'pdf': round(pdf, 4),
        'mean': round(mean, 4),
        'variance': round(variance, 4),
    }