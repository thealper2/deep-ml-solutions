from collections import deque


def create_replay_buffer(capacity):
    """Return an empty replay buffer with a fixed maximum capacity."""
    return {
        'data': deque(maxlen=capacity),
        'capacity': capacity
    }
