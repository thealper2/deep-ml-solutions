def compute_length_penalty(sequence_length, alpha):
    return ((5 + sequence_length) ** alpha) / (6 ** alpha)
