import numpy as np

class ReplayBuffer:
    """
    A fixed-size circular buffer to store and sample experience tuples
    for off-policy reinforcement learning.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        state = np.array(state)
        next_state = np.array(next_state)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, seed: int = None) -> dict:
        """
        Randomly sample a batch of transitions without replacement.
        Returns dict with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }
    
    def size(self) -> int:
        """Return current number of stored transitions."""
        return len(self.buffer)
