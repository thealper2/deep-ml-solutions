import numpy as np

def build_attention_mask(token_ids: np.ndarray, pad_token_id: int) -> np.ndarray:
    """
    Build a binary attention mask from padded token ids.

    Args:
        token_ids: array of shape (batch_size, seq_len)
        pad_token_id: integer id used for padding

    Returns:
        Integer mask array of shape (batch_size, seq_len) with 1 for real
        tokens and 0 for padding positions.
    """
    padded = np.where(token_ids != pad_token_id, 1, 0)
    return padded
