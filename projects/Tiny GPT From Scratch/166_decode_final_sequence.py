def decode_final_sequence(generated_ids, itos):
    """Decode a (1, T) id tensor into a string using itos."""
    return ''.join([itos[char] for char in generated_ids[0, :]])
