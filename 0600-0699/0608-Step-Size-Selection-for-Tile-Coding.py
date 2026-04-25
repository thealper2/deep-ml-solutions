import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    adjusted_alpha = base_alpha / n_tilings
    prediction_before = sum(weights[i] for i in active_tiles)
    error = target - prediction_before
    updated_weights = weights.copy()
    for idx in active_tiles:
        updated_weights[idx] += adjusted_alpha * error

    prediction_after = sum(updated_weights[i] for i in active_tiles)

    adjusted_alpha = round(adjusted_alpha, 4)
    prediction_before = round(prediction_before, 4)
    prediction_after = round(prediction_after, 4)
    updated_weights = [round(w, 4) for w in updated_weights]

    return {
        "adjusted_alpha": adjusted_alpha,
        "prediction_before": prediction_before,
        "prediction_after": prediction_after,
        "updated_weights": updated_weights,
    }