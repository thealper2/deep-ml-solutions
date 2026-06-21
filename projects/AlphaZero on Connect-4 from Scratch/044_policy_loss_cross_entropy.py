import torch

def policy_loss_cross_entropy(predicted_log_probs, target_policy):
    """Cross-entropy between MCTS target policy and network log-probs. Returns scalar tensor."""
    loss = -torch.sum(target_policy * predicted_log_probs, dim=1)
    return torch.mean(loss)
