def tokenize_corpus(texts: list) -> list:
    return [tokenize(clean_text(text)) for text in texts]
