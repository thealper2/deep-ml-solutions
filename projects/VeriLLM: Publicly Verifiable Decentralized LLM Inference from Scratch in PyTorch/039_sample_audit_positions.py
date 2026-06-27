def sample_audit_positions(seed, num_steps, k):
    if k == 0:
        return []

    if k >= num_steps:
        return list(range(num_steps))

    rng = np.random.default_rng(seed)
    indices = rng.choice(num_steps, size=k, replace=False)
    return sorted(indices.tolist())
