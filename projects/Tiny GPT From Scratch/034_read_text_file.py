def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    if not isinstance(text_blob, str):
        raise TypeError('Input must be a string')
    if len(text_blob) == 0:
        raise ValueError('Input string cannot be empty')
    return text_blob
