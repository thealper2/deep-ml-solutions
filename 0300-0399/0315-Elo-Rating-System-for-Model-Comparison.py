
import numpy as np

def elo_rating_update(ratings: dict, matches: list, k_factor: float) -> dict:
    """
    Update Elo ratings based on pairwise comparison results.
    
    Args:
        ratings: Dictionary mapping model names to their current Elo ratings
        matches: List of tuples (model_a, model_b, result) where result is 'a', 'b', or 'draw'
        k_factor: The K-factor controlling rating update magnitude
    
    Returns:
        Dictionary with updated ratings for all models
    """
    expected_score = lambda ra, rb: 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for model_a, model_b, result in matches:
        ra = ratings[model_a]
        rb = ratings[model_b]

        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)

        if result == 'a':
            sa, sb = 1.0, 0.0
        elif result == 'b':
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ratings[model_a] = ra + k_factor * (sa - ea)
        ratings[model_b] = rb + k_factor * (sb - eb)

    return ratings
