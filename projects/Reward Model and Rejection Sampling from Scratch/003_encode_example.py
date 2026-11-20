LETTERS = "abcdefgh"
CHARS = LETTERS + "#:"
STOI = {ch: i for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)


def encode_example(prompt, completion=""):
    """Template text "#" + prompt + ":" + completion -> (ids, completion_mask)."""
    text = "#" + prompt + ":" + completion
    ids = [STOI[ch] for ch in text]
    mask = [0] * (1 + len(prompt) + 1) + [1] * len(completion)
    return ids, mask