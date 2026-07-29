def train_tokenizer(corpus, vocab_size):
    """
    Build a tokenizer from the training corpus for downstream language modeling.

    The harness will use your tokenizer to train a bigram language model.
    Better tokenizers create more predictable token sequences, leading to
    lower bits-per-character on held-out data.

    Args:
        corpus: list[str] -- training documents (names like 'emma', 'olivia', ...)
        vocab_size: int   -- maximum vocabulary size (256)

    Returns:
        encode: callable, str -> list[int]
            Convert text to token IDs. All IDs must be in [0, vocab_size).
        decode: callable, list[int] -> str
            Convert token IDs back to text. Must satisfy decode(encode(s)) == s.
    """
def train_tokenizer(corpus, vocab_size):
    char_counts = {}
    for name in corpus:
        for c in name:
            char_counts[c] = char_counts.get(c, 0) + 1
    
    bigram_counts = {}
    for name in corpus:
        for i in range(len(name) - 1):
            bg = name[i:i+2]
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
    
    sorted_bigrams = sorted(bigram_counts.items(), key=lambda x: (-x[1], x[0]))
    
    vocab = {}
    for c in sorted(char_counts.keys()):
        vocab[c] = len(vocab)
    
    for bg, _ in sorted_bigrams:
        if len(vocab) >= vocab_size:
            break
        if bg not in vocab:
            vocab[bg] = len(vocab)
    
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    tokens_by_len = sorted(vocab.keys(), key=len, reverse=True)
    vocab_set = set(vocab.keys())
    
    def encode(text):
        result = []
        i = 0
        while i < len(text):
            matched = False
            for token in tokens_by_len:
                if text[i:i+len(token)] == token:
                    result.append(vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                c = text[i]
                if c in vocab:
                    result.append(vocab[c])
                else:
                    result.append(0)
                i += 1
        return result
    
    def decode(ids):
        return ''.join(reverse_vocab[idx] for idx in ids)
    
    return encode, decode
