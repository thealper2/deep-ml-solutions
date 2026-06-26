import numpy as np

def compute_target_q_with_target_network(target_params, batch, gamma):
    """Compute DQN bootstrap targets r + gamma * max_a' Q_target(s', a')."""
    next_states = batch['next_states']
    rewards = batch['rewards']
    dones = batch['dones']
    next_legal_masks = batch['next_legal_masks']
    batch_size = next_states.shape[0]
    
    q_next, _ = mlp_forward_pass(target_params, next_states)
    q_masked = np.where(next_legal_masks, q_next, -np.inf)
    max_q = np.max(q_masked, axis=1)
    max_q = np.where(np.isinf(max_q), 0.0, max_q)
    targets = rewards + gamma * max_q * (~dones)
    
    return targets
