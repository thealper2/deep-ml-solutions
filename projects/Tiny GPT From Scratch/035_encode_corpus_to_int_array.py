def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    return np.array([stoi[char] for char in text], dtype=np.int64)
