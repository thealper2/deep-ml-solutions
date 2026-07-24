import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.

    Args:
        chunk_sizes:   tokens per chunk, in order
        current_chunk: index of the chunk being generated

    Returns:
        Boolean array of shape (total_tokens, total_tokens).
        True means the row token can attend to the column token.
    """
    total_tokens = sum(chunk_sizes)
    mask = np.zeros((total_tokens, total_tokens), dtype=bool)

    chunk_starts = [0]
    for size in chunk_sizes:
        chunk_starts.append(chunk_starts[-1] + size)

    history_end = chunk_starts[current_chunk]
    current_start = chunk_starts[current_chunk]
    current_end = chunk_starts[current_chunk + 1]

    for i in range(history_end):
        mask[i, :history_end] = True

    for i in range(current_start, current_end):
        mask[i, :history_end] = True
        mask[i, current_start:i+1] = True

    return mask
