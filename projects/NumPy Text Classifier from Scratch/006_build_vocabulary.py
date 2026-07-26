def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    top_words = sorted_words[:max_size]
    vocab = {word: idx for idx, (word, _) in enumerate(top_words)}
    return vocab
