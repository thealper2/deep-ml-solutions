import os
import tempfile
import urllib.request
import torch

def load_data(train_frac=0.9):
    """Return (chars, encode, decode, train_data, val_data) for Tiny Shakespeare, one id per character."""
    path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return chars, encode, decode, data[:n], data[n:]