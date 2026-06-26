import numpy as np


def sample_minibatch_from_buffer(buffer, batch_size, rng):
    """Draw `batch_size` random transitions from `buffer` and stack fields into arrays."""
    data = buffer['data']
    indices = rng.integers(0, len(data), size=batch_size)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    next_legal_masks = []

    for idx in indices:
        transition = data[idx]
        states.append(transition['state'])
        actions.append(transition['action'])
        rewards.append(transition['reward'])
        next_states.append(transition['next_state'])
        dones.append(transition['done'])
        next_legal_masks.append(transition['next_legal_mask'])

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_states': np.array(next_states),
        'dones': np.array(dones),
        'next_legal_masks': np.array(next_legal_masks)
    }
