import numpy as np

def broadcast_turn_to_tokens(turn_advantages, turn_ids):
    """out[t] = turn_advantages[turn_ids[t]]."""
    return np.array(turn_advantages)[turn_ids]
