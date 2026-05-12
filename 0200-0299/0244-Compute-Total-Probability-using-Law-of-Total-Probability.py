def law_of_total_probability(priors: dict, conditionals: dict) -> float:
    """
    Compute P(A) using the Law of Total Probability.
    
    Args:
        priors: Dictionary mapping partition event names to P(Bi)
        conditionals: Dictionary mapping partition event names to P(A|Bi)
    
    Returns:
        float: The total probability P(A), rounded to 4 decimal places
    """
    total = 0.0
    for event in priors:
        total += priors[event] * conditionals[event]
    
    return round(total, 4)