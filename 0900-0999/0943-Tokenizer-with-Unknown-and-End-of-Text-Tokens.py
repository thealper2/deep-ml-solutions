import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == 'encode':
        pattern = r'([,.:;?_!"()\']|--|\s)'
        parts = re.split(pattern, text)
        tokens = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                tokens.append(stripped)

        result = []
        unk_id = vocab['<|unk|>']
        for token in tokens:
            if token in vocab:
                result.append(vocab[token])
            else:
                result.append(unk_id)

        return result

    elif mode == 'decode':
        reverse_vocab = {v: k for k, v in vocab.items()}
        tokens = [reverse_vocab[token_id] for token_id in text]
        decoded = ' '.join(tokens)
        pattern = r'\s+([,.:;?_!"()\'])'
        decoded = re.sub(pattern, r'\1', decoded)
        return decoded

    else:
        raise ValueError('Invalid mode')