import re

def encode(text, vocab):
    pattern = r'([,.:;?_!"()\']|--|\s)'
    parts = re.split(pattern, text)
    tokens = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            tokens.append(stripped)

    return [vocab[token] for token in tokens]

def decode(ids, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    tokens = [reverse_vocab[i] for i in ids]
    text = ' '.join(tokens)
    pattern = r'\s+([,\.\?\!\"\(\)\'])'
    text = re.sub(pattern, r'\1', text)
    return text