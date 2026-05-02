import numpy as np

def temporal_aggregate(features, k: int):
    """
    Aggregate frame features by averaging every k consecutive frames.

    Args:
        features: 2D array-like of shape (num_frames, feature_dim)
        k: aggregation factor (positive int)

    Returns:
        Nested list of shape (num_frames // k, feature_dim)
    """
    features = np.array(features)
    num_frames, feature_dim = features.shape
    num_groups = num_frames // k

    if num_groups == 0:
        return []

    features_truncated = features[:num_groups * k]
    grouped = features_truncated.reshape(num_groups, k, feature_dim)
    aggregated = np.mean(grouped, axis=1)
    return aggregated.tolist()