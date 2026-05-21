import numpy as np

def predict_last_token(logits: np.ndarray) -> list:
    """
    Extract last-token logits and return predicted class labels.

    Args:
        logits: array of shape (batch_size, seq_len, num_classes)

    Returns:
        List of predicted class labels (one int per batch element).
    """
    last_token_logits = logits[:, -1, :]
    predicted_class_ids = np.argmax(last_token_logits, axis=-1)
    return predicted_class_ids