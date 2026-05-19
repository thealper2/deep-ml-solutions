def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    pairs = []
    idx = 0
    while idx + max_length < len(token_ids):
        input_chunk = token_ids[idx:idx+max_length]
        target_chunk = token_ids[idx+1:idx+max_length+1]
        pairs.append((input_chunk, target_chunk))
        idx += stride

    return pairs