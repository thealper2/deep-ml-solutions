def sample_preference_batch(pairs, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_pairs = len(pairs)
    indices = rng.choice(n_pairs, size=batch_size, replace=(batch_size > n_pairs))
    batch = {}
    keys = ['chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask']
    if 'prompt' in pairs[0]:
        keys.append('prompt')

    for key in keys:
        batch[key] = np.array([pairs[i][key] for i in indices])

    return batch
