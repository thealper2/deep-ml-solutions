import numpy as np

def encode_prompt(prompt, stoi):
    """Encode a string prompt to an int ndarray of shape (1, T)."""
    T = len(prompt)
    encoded = np.array([stoi[char] for char in prompt]).reshape(1, T)
    return encoded
