def apply_random_walk_drift(true_values, drift_std, rng):
    noise = rng.normal(loc=0.0, scale=drift_std, size=true_values.shape)
    return true_values + noise
