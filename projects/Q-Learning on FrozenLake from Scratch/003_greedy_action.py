import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    return int(np.argmax(q_table[state]))
