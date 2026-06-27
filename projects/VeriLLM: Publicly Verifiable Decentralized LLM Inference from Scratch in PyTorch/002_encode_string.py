def encode_string(text, vocab):
    return [vocab['stoi'][c] for c in text]
