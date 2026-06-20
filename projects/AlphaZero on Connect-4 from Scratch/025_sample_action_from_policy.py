import torch

def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""
    masked = masked_policy_logits(logits, mask)
    scaled = masked / temperature
    probs = torch.softmax(scaled, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    return action
