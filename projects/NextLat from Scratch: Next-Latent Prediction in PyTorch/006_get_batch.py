def get_batch(dataset: dict, batch_size: int, step: int) -> dict:
    n = dataset['tokens'].shape[0]
    T = dataset['tokens'].shape[1]

    indices = [(step * batch_size + i) % n for i in range(batch_size)]

    tokens = dataset['tokens'][indices]
    mask = dataset['mask'][indices]
    states = dataset['states'][indices]

    x = tokens[:, :-1]
    y = tokens[:, 1:]
    mask_shifted = mask[:, 1:]
    states_shifted = states[:, :-1]

    return {
        'x': x,
        'y': y,
        'mask': mask_shifted,
        'states': states_shifted,
    }