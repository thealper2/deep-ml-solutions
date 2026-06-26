def append_transition_to_buffer(buffer, state, action, reward, next_state, done, next_legal_mask):
    """Append one (s, a, r, s', done, next_legal_mask) transition to the replay buffer."""
    buffer['data'].append((state, action, reward, next_state, done, next_legal_mask))
    return buffer
