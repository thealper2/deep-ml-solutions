import numpy as np

def map_latent_to_real(latents_labeled: np.ndarray, reals_labeled: np.ndarray, latents_query: np.ndarray) -> list:
    """Fit a linear map from latent actions to real actions using the labeled set,
    then apply it to latents_query. Return predictions as a nested list."""
    W = np.linalg.lstsq(latents_labeled, reals_labeled, rcond=None)[0]
    predictions = latents_query @ W
    return predictions.tolist()