import numpy as np

def gather_last_token_logits(logits: np.ndarray, input_ids: np.ndarray, pad_token_id: int) -> np.ndarray:
    """
    Gather the logit vector at the last non-padding token position for each sequence.

    logits: array of shape (B, T, V)
    input_ids: array of shape (B, T)
    pad_token_id: int

    Returns: array of shape (B, V)
    """
    B, T, V = logits.shape
    result = np.zeros((B, V))
    for b in range(B):
        non_pad_positions = np.where(input_ids[b] != pad_token_id)[0]
        if len(non_pad_positions) == 0:
            last_pos = 0
        else:
            last_pos = non_pad_positions[-1]

        result[b] = logits[b, last_pos, :]

    return result
