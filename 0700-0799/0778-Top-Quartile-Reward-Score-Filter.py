import numpy as np

def top_quartile_mask(scores: list) -> list:
    """
    Return a boolean list marking scores in the top quartile (>= 75th percentile).
    """
    if not scores:
        return []
        
    scores = np.array(scores)
    q3 = np.quantile(scores, 0.75)
    result = scores >= q3
    return result.tolist()