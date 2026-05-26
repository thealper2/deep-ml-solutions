import numpy as np

def compute_log_prob_ratios(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    mask: np.ndarray
) -> dict:
    """
    Compute log-probability ratios between policy and reference models.

    Args:
        policy_log_probs: (B, T) log-probs under the policy model
        reference_log_probs: (B, T) log-probs under the reference model
        mask: (B, T) binary mask, 1 for valid tokens, 0 for padding

    Returns:
        Dictionary with:
          - 'per_token_log_ratios': (B, T) array
          - 'sequence_log_ratios': (B,) array
          - 'mean_sequence_log_ratio': float scalar
    """
    per_token_log_ratios = (policy_log_probs - reference_log_probs) * mask
    sequence_log_ratios = np.sum(per_token_log_ratios, axis=1)
    mean_sequence_log_ratio = np.mean(sequence_log_ratios) 

    return {
        'per_token_log_ratios': per_token_log_ratios,
        'sequence_log_ratios': sequence_log_ratios,
        'mean_sequence_log_ratio': mean_sequence_log_ratio,
    }